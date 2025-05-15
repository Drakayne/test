from flask import Flask
import threading
import time
import os
from datetime import datetime, timedelta
import schedule
import logging
import sys

# Import your data collection script
import data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scheduler.log')
    ]
)
logger = logging.getLogger('scheduler')

# Initialize Flask app
app = Flask(__name__)

# Track the last run and next scheduled run
last_run = None
next_run = None

def update_next_run():
    """Update the next scheduled run time"""
    global next_run
    next_run = datetime.now() + timedelta(hours=8)
    logger.info(f"Next data collection scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

def run_data_collection():
    """Run the data collection script and update schedule"""
    global last_run
    logger.info("Starting scheduled data collection...")
    
    try:
        # Run the data collection script
        result = data.main()
        last_run = datetime.now()
        
        if result:
            logger.info(f"Data collection completed successfully at {last_run.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logger.error("Data collection returned failure status")
            
        # Update next run time
        update_next_run()
        
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        import traceback
        logger.error(traceback.format_exc())

def run_threaded(job_func):
    """Run the function in a separate thread"""
    job_thread = threading.Thread(target=job_func)
    job_thread.start()

@app.route('/')
def home():
    """Website home page - status dashboard"""
    current_time = datetime.now()
    
    # Check if it's time for a manual trigger (>8 hours since last run)
    if last_run and current_time - last_run > timedelta(hours=8):
        logger.info("Manual trigger detected via web request")
        # Start collection in a separate thread to not block the response
        threading.Thread(target=run_data_collection).start()
    
    # Get directory listing for data files
    try:
        data_dir = 'funding_arb_data'
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            files.sort(reverse=True)
            recent_files = files[:10]  # Get 10 most recent files
        else:
            recent_files = []
    except Exception as e:
        recent_files = [f"Error listing files: {str(e)}"]
    
    # Format time info
    last_run_str = last_run.strftime('%Y-%m-%d %H:%M:%S') if last_run else "Never"
    next_run_str = next_run.strftime('%Y-%m-%d %H:%M:%S') if next_run else "Not scheduled"
    
    time_since_last = current_time - last_run if last_run else timedelta(days=9999)
    time_until_next = next_run - current_time if next_run else timedelta(seconds=0)
    
    # Build HTML response
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Funding Data Collection</title>
        <meta http-equiv="refresh" content="300"> <!-- Refresh every 5 minutes -->
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1 {{ color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .status {{ margin: 20px 0; padding: 15px; border-radius: 5px; }}
            .running {{ background-color: #d4edda; color: #155724; }}
            .scheduled {{ background-color: #d1ecf1; color: #0c5460; }}
            .files {{ margin-top: 20px; background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            .file-list {{ list-style-type: none; padding-left: 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Crypto Funding Data Collection</h1>
            
            <div class="status running">
                <strong>Status:</strong> Running<br>
                <strong>Last collection:</strong> {last_run_str} ({time_since_last.total_seconds() / 3600:.1f} hours ago)<br>
                <strong>Next collection:</strong> {next_run_str} (in {time_until_next.total_seconds() / 3600:.1f} hours)
            </div>
            
            <div class="files">
                <h3>Recent Data Files:</h3>
                <ul class="file-list">
    """
    
    # Add file list
    for file in recent_files:
        html += f"<li>{file}</li>"
    
    if not recent_files:
        html += "<li>No data files found</li>"
    
    html += """
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def setup_schedule():
    """Set up the scheduler for regular data collection"""
    # Clear any existing schedules
    schedule.clear()
    
    # Schedule to run every 8 hours
    schedule.every(8).hours.do(run_threaded, run_data_collection)
    logger.info("Scheduled data collection to run every 8 hours")
    
    # Update next run time
    update_next_run()

def scheduler_thread():
    """Thread that runs the scheduler"""
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in scheduler thread: {e}")
            time.sleep(300)  # Sleep for 5 minutes on error

if __name__ == "__main__":
    # Run initial data collection
    logger.info("Starting initial data collection...")
    run_data_collection()
    
    # Set up the scheduler
    setup_schedule()
    
    # Start the scheduler in a separate thread
    threading.Thread(target=scheduler_thread, daemon=True).start()
    
    # Start the Flask server (this keeps the repl alive)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
