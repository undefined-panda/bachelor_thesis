"""
@author Jad Dayoub, 7425569
"""

import os
import subprocess

def create_requirements_txt():
    """
    Creates 'requirements.txt'-file in superior folder based on used modules in project.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    os.chdir(project_dir)
    
    try:
        subprocess.run(['pipreqs', '.', '--force'], check=True)
        print("'requirements.txt'-file created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while creating `requirements.txt`-file: {e}")

create_requirements_txt()