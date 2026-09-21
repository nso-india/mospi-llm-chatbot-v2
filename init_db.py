from models import init_db

# Remove asyncio.run()
async def initialize_database():
    await init_db()
