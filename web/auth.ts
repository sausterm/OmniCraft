import NextAuth from "next-auth";
import { DrizzleAdapter } from "@auth/drizzle-adapter";
import Resend from "next-auth/providers/resend";
import { db } from "@/lib/db";

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: DrizzleAdapter(db),
  providers: [
    Resend({
      from: process.env.EMAIL_FROM || "Artisan <onboarding@resend.dev>",
    }),
  ],
  pages: {
    signIn: "/login",
    verifyRequest: "/check-email",
  },
  callbacks: {
    session({ session, user }) {
      if (session.user) {
        session.user.id = user.id;
      }
      return session;
    },
  },
  theme: {
    colorScheme: "light",
    brandColor: "#059669",
    logo: "/logo.png",
  },
});
