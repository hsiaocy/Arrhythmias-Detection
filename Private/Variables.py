"""
Variables

- group1: cases from #100 to #124
- group2: cases from #200 to #234
- all_cases: all cases include Paced cases {'102', '104', '107', '217'}
- experiment_cases: all cases exclude Paced cased for my experiemnt
- ann5 = 5 annotation types(Normal, PVC, Paced, LBB, RBB)

"""


group1 = ['100', '101', '103', '105', '106', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124']
group2 = ['200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
all_cases = group1 + group2 + ['102', '104', '107', '217']
experiment_cases = group1 + group2
ann5 = ['N', 'V', '/', 'L', 'R']
