"""
Training Statics Tools

A class for loading statistics related to a particular rutraiining session.
"""

import numpy as np
#from scipy import stats
import pandas as pd
import os

def str_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

def is_stat_file_version(file_name, version):
    return file_name.startswith("stats_{}_gen".format(version)) and file_name.endswith(".h5")

class TrainingStates:
    def __init__(self, versions, directory, verbose=True):
        self.stats_files = self.get_stat_files(versions, directory)
        
        if verbose:
            print("Loading files:")
            for f in self.stats_files:
                print(directory + f)

        self.generation_stats = self.load_stats('generation_stats')
        self.game_stats = self.load_stats('game_stats')
        self.move_stats = self.load_stats('self_play_stats')

    def get_stat_files(self, versions, directory):
        stat_files = []
        for version in reversed(versions):
            files = [directory + f for f in os.listdir(directory) if is_stat_file_version(f, version)]
            stat_files += list(sorted(files))

        return stat_files

    def load_stats(self, key_name):
        df_list = []
        for f in self.stats_files:
            path = f
            generation = str_between(f, "_gen", ".h5")
            df = pd.read_hdf(path, key=key_name)
            df['_generation'] = int(generation)
            df_list.append(df)

        if df_list:
            stats = pd.concat(df_list, ignore_index=True)
        else:
            return pd.DataFrame()
            
        return stats

    def first_move_stats(self):
        """
        Note: There is an indexing issue (the index of first_play_stats is the orginal index
        while the index of game_stats is the game number).  The easiest fix is to just use
        the values (an array) of the series and not the series itself.
        """
        return self.move_stats[self.move_stats['_step_id'] == 0]

    def found_target_on_first_move(self):
        return (self.first_move_stats()['shortest_path'] >= 0).values

    def lost_but_found_target_on_first_move(self):
        return self.found_target_on_first_move() & ~self.game_stats['win']

    def win_but_did_not_find_target_on_first_move(self):
        return ~self.found_target_on_first_move() & self.game_stats['win']

if __name__ == '__main__':
    from pprint import pprint
    versions = ['v0.9.3']
    save_dir = '../save/stats_v0.9.3/'
    #VERSIONS = ['v0.9.2.1', 'v0.9.2']
    #SAVE_DIR = '../save/stats_archive/'

    cube_stats = TrainingStates(versions, save_dir)

    pprint(cube_stats.generation_stats)

    pprint(np.mean(cube_stats.lost_but_found_target_on_first_move()))
    pprint(np.mean(cube_stats.win_but_did_not_find_target_on_first_move()))



