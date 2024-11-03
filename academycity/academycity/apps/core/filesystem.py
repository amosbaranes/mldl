import shutil
import os


class FS(object):
    def __init__(self):
        self.path = None

    def get_info_for(self, path="/home/User/Documents"):
        self.path = path
        # Get the disk usage statistics
        # about the given path
        stat = shutil.disk_usage(self.path)
        data = {}
        data['total'] = round(stat.total/2**30,2)
        data['used'] = round(stat.used/2**30,2)
        data['free'] = round(stat.free/2**30,2)
        return data

    def get_files_in(self, path="/home/User/Documents"):
        self.path = path
        files = []
        for basename in os.listdir(path):
            filename = os.path.join(path, basename)
            if os.path.isfile(filename):
                files.append((basename, filename))

        # Re-populate list with filename, size tuples
        for i in iter(range((len(files)))):
            files[i] = (files[i][0], os.path.getsize(files[i][1]))

        # Sort list by file size
        # If reverse=True sort from largest to smallest
        # If reverse=False sort from smallest to largest
        files.sort(key=lambda filename: filename[1])

        # Re-populate list with just filenames
        # for i in os.listdir(len(files)):
        #     filepaths[i] = files[i][0]

        return files







        files = os.listdir(path)
        return files
