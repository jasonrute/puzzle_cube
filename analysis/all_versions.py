"""
Find all version numbers and directories
"""

import os

def str_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

def is_stat_file_version(file_name, version):
    return file_name.startswith("stats_{}_gen".format(version)) and file_name.endswith(".h5")

def version(file_name):
    return str_between(file_name, "stats_", "_gen")
       
def all_versions(directories):
    versions = set() 
    for d in directories:
        for f in os.listdir(d):
            if f.startswith("stats"):
                v = version(f)
                f_dir = d
                versions.add((v, f_dir))
            elif not f.startswith(".") and not f.endswith(".h5") and not f.endswith(".tar") and not f.endswith(".gz"):
                v = f
                f_dir = d + f + '/'
                if any(f.startswith("stats") for f in os.listdir(f_dir)):
                    versions.add((v, f_dir))
    versions = [(v, d) for v, d in versions if v.startswith('v') and v >= 'v0.4']
    versions = sorted(list(versions))
    return versions

if __name__ == '__main__':
    from pprint import pprint
    directories = ['../../cube_nn_mcts/save/', '../save/', '../results/', '../aws_save/', '../aws_results/']
    pprint(all_versions(directories))




