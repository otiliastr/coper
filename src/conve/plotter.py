import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pylab

# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)
#
# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()


def get_results_from_file(file_path):
    iterations = []
    results = []
    with open(file_path, 'r') as file:
        all_results = file.readlines()
        for iteration, eval_result in enumerate(all_results):
          iterations.append(iteration * 5)
          results.append(float(eval_result))
    return iterations, results

def plot_hits_comparisons(file_prefix, comp_type= 'overall'):
    if comp_type not in ['overall', 'left', 'right']:
        print("Please specify the correct comparison type: overall, left, or right")
    else:
        comparison_type = '' if comp_type == 'overall' else comp_type + "_"

        path_hits_at_1 = file_prefix + 'dev_evaluationhits_{0}at_1.txt'.format(comparison_type)
        path_hits_at_3 = file_prefix + 'dev_evaluationhits_{0}at_3.txt'.format(comparison_type)
        path_hits_at_10 = file_prefix + 'dev_evaluationhits_{0}at_10.txt'.format(comparison_type)

        hits_1_iterations, hits_1_results = get_results_from_file(path_hits_at_1)
        hits_3_iterations, hits_3_results = get_results_from_file(path_hits_at_3)
        hits_10_iterations, hits_10_results = get_results_from_file(path_hits_at_10)

        pylab.plot(hits_10_iterations, hits_10_results, 'g-', label='Hits @10')
        pylab.plot(hits_3_iterations, hits_3_results, 'b-', label='Hits @3')
        pylab.plot(hits_1_iterations, hits_1_results, 'r-', label = 'Hits @1')

        pylab.legend(loc='lower right')
        pylab.xlabel("Iteration")
        pylab.ylabel("Accuracy ([0., 1.] scale)")
        pylab.title("Accuracy over iterations")
        pylab.show()

# make this take any number of args
def plot_comparisons(plot_info = {}, *args):
    plot_data = {}
    for file_idx in range(len(args)):
        file = args[file_idx]
        iterations, results = get_results_from_file(file)
        pylab.plot(iterations, results, label = plot_info['legend_label_{}'.format(file_idx+1)])
    pylab.legend(loc='lower right')
    pylab.xlabel('Epoch')
    pylab.ylabel('Performance')
    pylab.title(plot_info['title'])
    pylab.xlim(xmin = 0, xmax=1000)
    pylab.show()



    # iterations_1, results_1 = get_results_from_file(file_1)
    # iterations_2, results_2 = get_results_from_file(file_2)
    # pylab.plot(iterations_1, results_1, 'b-', label = plot_info['legend_label_1'])
    # pylab.plot(iterations_2, results_2, 'r-', label = plot_info['legend_label_2'])
    # pylab.legend(loc='lower right')
    # pylab.xlabel('Iteration')
    # pylab.ylabel('Performance')
    # pylab.title(plot_info['title'])
    # pylab.xlim(xmax=200)
    # pylab.show()

# def plot_comparisons(plot_info = {}, file_1= '', file_2 = ''):
#     iterations_1, results_1 = get_results_from_file(file_1)
#     iterations_2, results_2 = get_results_from_file(file_2)
#     pylab.plot(iterations_1, results_1, '-', label = plot_info['legend_label_1'])
#     pylab.plot(iterations_2, results_2, '-', label = plot_info['legend_label_2'])
#     pylab.legend(loc='lower right')
#     pylab.xlabel('Iteration')
#     pylab.ylabel('Performance')
#     pylab.title(plot_info['title'])
#     pylab.xlim(xmax=200)
#     pylab.show()

# dev_path_hits_at_1 = '/Users/gstoica/Desktop/experiments/logs/conve_baseline_opt_bl_params/dev_evaluationhits_at_1.txt'
# dev_path_hits_at_3 = '/Users/georgestoica/Desktop/Research/QA/ConvE/logs/dev_evaluationhits_at_3.txt'
# dev_path_hits_at_10 = '/Users/georgestoica/Desktop/Research/QA/ConvE/logs/dev_evaluationhits_at_10.txt'

# test_path = '/Users/georgestoica/Desktop/Research/QA/ConvE/logs/test_evaluationhits_at_1.txt'
# hits_1_iterations, hits_1_results = get_results_from_file(dev_path_hits_at_1)
# hits_3_iterations, hits_3_results = get_results_from_file(dev_path_hits_at_3)
# hits_10_iterations, hits_10_results = get_results_from_file(dev_path_hits_at_10)
# test_iterations, test_results = get_results_from_file(test_path)

# fig, ax = plt.subplots()
# ax.plot(hits_1_iterations, hits_1_results, 'r--')
# ax.plot(hits_3_iterations, hits_3_results, 'b--')
# ax.plot(hits_10_iterations, hits_10_results, 'g--')
# ax.plot(test_iterations, test_results, 'b--')
# plt.show()

# pylab.plot(hits_1_iterations, hits_1_results, 'r--', label = 'Hits @1')
# pylab.plot(hits_3_iterations, hits_3_results, 'b--', label = 'Hits @3')
# pylab.plot(hits_10_iterations, hits_10_results, 'g--', label = 'Hits @10')
# pylab.legend(loc='lower right')
# pylab.xlabel("Iteration")
# pylab.ylabel("Accuracy ([0., 1.] scale)")
# pylab.title("Accuracy over iterations")
# pylab.show()

# plot_hits_comparisons(file_prefix='/Users/gstoica/Desktop/experiments/logs/conve_baseline_opt_bl_params/',
#                       comp_type='overall')

# plot_hits_comparisons(file_prefix='/Users/gstoica/Desktop/experiments/logs/conve_equal_merge_opt_bl_params/',
#                       comp_type='overall')

plot_info = {'title': 'Optimized Baseline vs Optimized Rel-Struc-ConvE \n Performances on Hits@1',
             'legend_label_1': 'Baseline',
             'legend_label_2': 'Rel-Struc-ConvE'}

baseline_file = '/Users/gstoica/Desktop/experiments/logs2/conve_baseline_opt_bl_params_2/test_evaluationhits_at_1.txt'
# struc_conve_2_file = '/Users/gstoica/Desktop/experiments/logs2/conve_equal_merge_opt_bl_params_test_2/test_evaluationmean_rank.txt'
struc_conve_1_file = '/Users/gstoica/Desktop/experiments/logs2/conve_equal_merge_opt_bl_params/test_evaluationhits_at_1.txt'
# struc_conve_lin_file = '/Users/gstoica/Desktop/experiments/logs2/conve_equal_merge_opt_bl_params_lin_infl_2/test_evaluationmean_rank.txt'
baseline_optimized = '/Users/gstoica/Desktop/experiments/logs2/conve_baseline_opt_bl_params_WN18RR/test_evaluationhits_at_1.txt'
opt_struc_file = '/Users/gstoica/Desktop/experiments/logs2/struc_rel_conve_opt_bl_params_WN18RR/test_evaluationhits_at_1.txt'

plot_comparisons(plot_info, baseline_optimized, opt_struc_file)
# plot_comparisons(plot_info, baseline_file, struc_conve_1_file)

