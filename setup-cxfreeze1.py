import sys
from cx_Freeze import setup, Executable
import os

packages = ['pkg_resources', 'scipy']
includefiles = [
            os.path.join(sys.base_prefix, 'Library', 'bin', 'sqlite3.dll'),
#            os.path.join(sys.base_prefix, 'Library', 'bin', 'mkl_intel_thread.dll'),
#            os.path.join(sys.base_prefix, 'Library', 'bin', 'mkl_core.dll'),
#            os.path.join(sys.base_prefix, 'Library', 'bin', 'mkl_def.dll'),
#            os.path.join(sys.base_prefix, 'Library', 'plugins/platforms'),            
            "cute_gui_r1.ui", "ApexCUTE2.ico", "ApexCUTE2.png", "add-file-48.ico"
			, "close-window-48.ico", "folder-48.ico", "save-48.ico", "save-as-48.ico"
        ]
excludes = ['scipy.spatial.cKDTree']


exe = Executable(
    script='main_prog.py',
    targetName='apex-cute.exe',
    base="Win32GUI"
    )

options ={
    'build_exe': {
        'packages': packages,
        'include_files': includefiles,
        'excludes': excludes,
        'build_exe': './/build'
    }
}

setup(
    name='apex-cute.exe',
    options=options,
    version='7.1',
    description='APEX-CUTE',
    executables=[exe]
    )
