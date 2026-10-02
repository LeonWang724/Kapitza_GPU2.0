# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:06:43 2026

@author: Alex
"""

import os


def update_config_value(file_path: str, identifier: str, new_value: str) -> None:
    """
    Updates a key-value pair in a simple configuration file.
    If the identifier doesn't exist, it appends it to the end of the file.
    """
    # Ensure new_value is a string just in case an int/float is passed
    new_value_str = str(new_value)
    
    try:
        # Read the existing content
        with open(file_path, 'r') as file:
            lines = file.readlines()

        key_found = False
        
        # Iterate through the lines to find the identifier
        for i, line in enumerate(lines):
            # Strip whitespace to check the start, but preserve original indentation
            stripped_line = line.strip()
            
            # Skip empty lines or standard comments
            if not stripped_line or stripped_line.startswith('#'):
                continue

            # Check if the line matches the identifier
            if stripped_line.startswith(f"{identifier}="):
                lines[i] = f"{identifier}={new_value_str}\n"
                key_found = True
                break

        # If the identifier wasn't found, append it to the end
        if not key_found:
            # Add a newline if the file didn't end with one
            if lines and not lines[-1].endswith('\n'):
                lines[-1] += '\n'
            lines.append(f"{identifier}={new_value_str}\n")

        # Write the updated lines back out
        with open(file_path, 'w') as file:
            file.writelines(lines)
            
        #print(f"Successfully updated '{identifier}' to '{new_value_str}' in {file_path}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")