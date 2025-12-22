import { drizzle } from "drizzle-orm/better-sqlite3";
import Database from "better-sqlite3";
import * as schema from "./schema";

// Database file location
const DB_PATH = process.env.DATABASE_PATH || "./data/artisan.db";

// Ensure data directory exists
import { mkdirSync } from "fs";
import { dirname } from "path";
try {
  mkdirSync(dirname(DB_PATH), { recursive: true });
} catch {
  // Directory exists
}

// Create SQLite connection
const sqlite = new Database(DB_PATH);

// Enable WAL mode for better performance
sqlite.pragma("journal_mode = WAL");

// Create Drizzle ORM instance
export const db = drizzle(sqlite, { schema });

// Initialize tables if they don't exist
sqlite.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT NOT NULL UNIQUE,
    emailVerified INTEGER,
    image TEXT,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );

  CREATE TABLE IF NOT EXISTS accounts (
    userId TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    provider TEXT NOT NULL,
    providerAccountId TEXT NOT NULL,
    refresh_token TEXT,
    access_token TEXT,
    expires_at INTEGER,
    token_type TEXT,
    scope TEXT,
    id_token TEXT,
    session_state TEXT,
    PRIMARY KEY (provider, providerAccountId)
  );

  CREATE TABLE IF NOT EXISTS sessions (
    sessionToken TEXT PRIMARY KEY,
    userId TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS verificationTokens (
    identifier TEXT NOT NULL,
    token TEXT NOT NULL,
    expires INTEGER NOT NULL,
    PRIMARY KEY (identifier, token)
  );

  CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    completed_at INTEGER
  );

  CREATE TABLE IF NOT EXISTS purchases (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    job_id TEXT NOT NULL REFERENCES jobs(id),
    product_id TEXT NOT NULL,
    email TEXT NOT NULL,
    stripe_session_id TEXT,
    is_promo INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );
`);

export { schema };
