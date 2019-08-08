import os

import config as cfg
import constants as const
import std_utils

def main():
    filepath = cfg.DATASET_DIR + '/' + cfg.FALSE_DETECTIONS_SUBDIR + '/' + \
                        'random_{}.txt'.format(cfg.RANDOM_MALICIOUS_PROBABILITY)

    with open(filepath, 'w') as f:
        for entity_str in const.valid_perspectives():
            if std_utils.decision_true(cfg.RANDOM_MALICIOUS_PROBABILITY):
                f.write('%s\n' % entity_str)

main()