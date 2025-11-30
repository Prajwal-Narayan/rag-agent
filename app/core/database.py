from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row  # 👈 NEW IMPORT
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.config import settings

# Configure the connection to return Dictionaries, not Tuples
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,  # 👈 THE FIX: Enforce DictRow format
}

pool = AsyncConnectionPool(
    conninfo=settings.DATABASE_URL,
    max_size=20,
    kwargs=connection_kwargs,
    open=False
)

checkpointer = None

async def init_db():
    """
    Initializes the DB pool and Checkpointer on startup.
    """
    global checkpointer
    
    await pool.open()
    
    # Instantiate Checkpointer
    # We ignore the type hint here because Pylance struggles with the
    # generic 'pool' vs 'connection' distinction, but it works at runtime.
    checkpointer = AsyncPostgresSaver(pool) # type: ignore
    
    async with pool.connection() as conn:
        await checkpointer.setup()
        print("✅ Postgres Checkpointer Tables Created.")