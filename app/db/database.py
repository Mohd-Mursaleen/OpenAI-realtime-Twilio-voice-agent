import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Supabase connection parameters
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Storage path for uploaded files
STORAGE_PATH = os.getenv("STORAGE_PATH", "storage")

# Initialize Supabase client
supabase: Client = None

def initialize_supabase():
    """Initialize the Supabase client if not already initialized."""
    global supabase
    if supabase is None:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

def get_supabase_client() -> Client:
    """Returns the Supabase client instance."""
    global supabase
    if supabase is None:
        supabase = initialize_supabase()
    return supabase

def get_storage_path() -> str:
    """
    Get the storage path for file uploads.
    Create the directory if it doesn't exist.
    
    Returns:
        str: Path to the storage directory
    """
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    return STORAGE_PATH 