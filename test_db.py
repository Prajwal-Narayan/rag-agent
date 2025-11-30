import asyncio
import selectors
from app.core.database import init_db

async def main():
    await init_db()

if __name__ == "__main__":
    # Force Psycopg to use a compatible loop on Windows
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

    asyncio.run(main())
