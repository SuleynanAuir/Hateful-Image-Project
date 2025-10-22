# log_utils.py

import datetime

# """
# 注意，每一步更新后，都得 ### 跑一遍下面的代码，然后自定义里面的更新内容:

# from log_utils import log_update
# log_update("自定义更新内容的中文描述...")

# """

# Define the path to your log file
log_file = "project_log.txt"

def log_update(message):
    """
    Logs a project update to a file with a timestamp.
    
    Parameters:
        message (str): The message to log, describing the update.
    
    """
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the log message with timestamp
    log_message = f"[{timestamp}] {message}\n"
    
    # Open the log file in append mode and write the message
    with open(log_file, "a") as file:
        file.write(log_message)

    print(f"Update logged: {log_message}")
