import { NextResponse } from "next/server";
import { auth } from "@/auth";
import { db, schema } from "@/lib/db";
import { eq, desc } from "drizzle-orm";
import { nanoid } from "nanoid";

// Credit costs for different actions
const CREDIT_COSTS = {
  generation: 10,      // Basic paint-by-numbers
  style_transfer: 20,  // Style transfer (GPU intensive)
  premium_export: 5,   // High-res PDF export
};

// Credit rewards for different actions
const CREDIT_REWARDS = {
  welcome: 100,        // New user signup
  referral: 50,        // Referred a friend who signed up
  referred: 25,        // Was referred by someone
  share: 10,           // Shared a creation
  daily_login: 5,      // Daily login bonus (once per day)
};

export async function GET() {
  const session = await auth();

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    // Get user's current credits
    const user = await db.query.users.findFirst({
      where: eq(schema.users.id, session.user.id),
      columns: { credits: true },
    });

    // Get recent transactions
    const transactions = await db.query.creditTransactions.findMany({
      where: eq(schema.creditTransactions.userId, session.user.id),
      orderBy: [desc(schema.creditTransactions.createdAt)],
      limit: 20,
    });

    return NextResponse.json({
      credits: user?.credits ?? 0,
      transactions: transactions.map(tx => ({
        id: tx.id,
        amount: tx.amount,
        type: tx.type,
        description: tx.description,
        createdAt: tx.createdAt,
      })),
      costs: CREDIT_COSTS,
      rewards: CREDIT_REWARDS,
    });
  } catch (error) {
    console.error("Error fetching credits:", error);
    return NextResponse.json({ error: "Failed to fetch credits" }, { status: 500 });
  }
}

// Spend credits
export async function POST(request: Request) {
  const session = await auth();

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body = await request.json();
    const { action, referenceId } = body;

    const cost = CREDIT_COSTS[action as keyof typeof CREDIT_COSTS];
    if (!cost) {
      return NextResponse.json({ error: "Invalid action" }, { status: 400 });
    }

    // Get current credits
    const user = await db.query.users.findFirst({
      where: eq(schema.users.id, session.user.id),
      columns: { credits: true },
    });

    if (!user || user.credits < cost) {
      return NextResponse.json({
        error: "Insufficient credits",
        required: cost,
        current: user?.credits ?? 0,
      }, { status: 402 });
    }

    // Deduct credits and create transaction
    await db.update(schema.users)
      .set({ credits: user.credits - cost })
      .where(eq(schema.users.id, session.user.id));

    await db.insert(schema.creditTransactions).values({
      id: `tx_${nanoid()}`,
      userId: session.user.id,
      amount: -cost,
      type: action,
      description: `Used ${cost} credits for ${action.replace('_', ' ')}`,
      referenceId: referenceId || null,
    });

    return NextResponse.json({
      success: true,
      creditsUsed: cost,
      remainingCredits: user.credits - cost,
    });
  } catch (error) {
    console.error("Error spending credits:", error);
    return NextResponse.json({ error: "Failed to spend credits" }, { status: 500 });
  }
}
