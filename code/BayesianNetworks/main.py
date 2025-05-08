import bnlearn as bn
import pandas as pd
import graphviz
from tabulate import tabulate
import os
import networkx as nx
from scipy.stats import chi2_contingency
import numpy as np
import copy

# record_dic = {}


def check_new_node(compare_list, current_node):
    new = True
    current_source = current_node[0]
    current_target = current_node[1]
    for compare_node in compare_list:
        compare_source = compare_node[0]
        compare_target = compare_node[1]
        if compare_source == current_source and compare_target == current_target:
            new = False
    return new


def compare_node(compare_list, current_list):
    compare_result = []
    for current_node in current_list:
        if check_new_node(compare_list, current_node):
            compare_result.append(1)
        else:
            current_source = current_node[0]
            current_target = current_node[1]
            current_value = current_node[3]
            for compare_node in compare_list:
                compare_source = compare_node[0]
                compare_target = compare_node[1]
                compare_value = compare_node[3]
                if current_source == compare_source and current_target == compare_target:
                    if current_value != compare_value:
                        compare_result.append(1)
                    else:
                        compare_result.append(0)
                else:
                    # source targe inconsistent
                    pass


def remove_node_without_label(dot):
    nodes_to_remove = []
    for node in dot.body:
        if 'label=' not in node:
            nodes_to_remove.append(node)

    # 从 body 中移除没有标签的节点
    for node in nodes_to_remove:
        dot.body.remove(node)
    return dot

def create_naive_bayes_figure(node_codeword, output_path, label, record):
    ## 有向图
    # DAG = bn.structure_learning.fit(data)
    # print(DAG)
    # graph = bn.plot(DAG)
    # print(graph)

    # print(os.environ["PATH"])
    # os.environ["PATH"] += ';' + 'C:\\Users\\zhyte\\AppData\\Local\\Temp\\GRAPHVIZ\\graphviz-2.38\\release\\bin'
    # print(os.environ["PATH"])
    # # 树状贝叶斯
    # model = bn.structure_learning.fit(data, methodtype='tan', class_node='L0')
    # G = bn.plot_graphviz(model)
    # G.view('1', output_path)
    # exit(0)
    # model1 = bn.independence_test(model, data, alpha=0.05, prune=False)
    # bn.plot(model1, pos=G['pos'])
    # print(tabulate(model1['independence_test'], headers="keys"))
    # model2 = bn.independence_test(model, data, alpha=0.05, prune=True)
    # # [bnlearn] >Edge [tub <-> asia] [P=0.104413] is excluded because it was not significant (P<0.05) with [chi_square]
    # # [bnlearn] >Edge [lung <-> asia] [P=1] is excluded because it was not significant (P<0.05) with [chi_square]
    # # [bnlearn] >Edge [lung <-> tub] [P=0.125939] is excluded because it was not significant (P<0.05) with [chi_square]
    # bn.plot(model2, pos=G['pos'])

    ## 天真贝叶斯 naive bayes
    # model = bn.structure_learning.fit(data, methodtype='naivebayes', root_node=node_codeword)
    # constrained
    # model = bn.structure_learning.fit(data, methodtype='cs')

    # TAN
    model = bn.structure_learning.fit(data, methodtype='tan', class_node=node_codeword)
    # CL
    # model = bn.structure_learning.fit(data, methodtype='cl')

    model_copy = copy.deepcopy(model)

    model_np = bn.independence_test(model_copy, data, prune=False, test='chi_square')




    # rewrite adjmat with chi_square values
    source = model_np.get('independence_test')['source']
    target = model_np.get('independence_test')['target']
    weights = []
    for i, j in model_np['model_edges']:
        contingency_table = pd.crosstab(data[i], data[j])
        chi, p_value, dof, expected = chi2_contingency(data.groupby([i, j]).size().unstack(j, fill_value=0),
                                                              lambda_="cressie-read" )
        n = contingency_table.sum().sum()
        v = np.sqrt(chi / (n * min(contingency_table.shape) - 1))
        weights.append(v)

    adjmat = bn.vec2adjmat(source, target, weights=weights, symmetric=True, aggfunc='sum', verbose=3)
    model_np['adjmat'] = adjmat

    NP_G = bn.plot_graphviz(model_np, edge_labels='None')
    file_1 = node_codeword + '_' + label + '_NP'
    NP_G.view(file_1, output_path)


    # Prune using the chi-square independence test
    model_wp = bn.independence_test(model_copy, data, prune=True)
    # print(model['independence_test'])
    # print(model.get('independence_test')['chi_square'].round(decimals=2))
    # [bnlearn] >Compute edge strength with [chi_square]
    # [bnlearn] >Edge [B <-> A] [P=0.240783] is excluded because it was not significant (P<0.05) with [chi_square]
    # [bnlearn] >Edge [B <-> C] [P=0.766384] is excluded because it was not significant (P<0.05) with [chi_square]
    # [bnlearn] >Edge [B <-> D] [P=0.382504] is excluded because it was not significant (P<0.05) with [chi_square]
    # Plot

    # rewrite adjmat with chi_square values
    source = model_wp.get('independence_test')['source']
    target = model_wp.get('independence_test')['target']
    weights = []
    for i, j in model_wp['model_edges']:
        contingency_table = pd.crosstab(data[i], data[j])
        chi, p_value, dof, expected = chi2_contingency(data.groupby([i, j]).size().unstack(j, fill_value=0),
                                                              lambda_="cressie-read" )
        n = contingency_table.sum().sum()
        v = np.sqrt(chi / (n * min(contingency_table.shape) - 1))
        weights.append(v)

    adjmat = bn.vec2adjmat(source, target, weights=weights, symmetric=True, aggfunc='sum', verbose=3)
    model_wp['adjmat'] = adjmat

    WP_G = bn.plot_graphviz(model_wp, edge_labels='None')
    file_2 = node_codeword + '_' + label + '_WP'
    WP_G.view(file_2, output_path)
    # for node in WP_G:
    #     print(node)
    # WP_G_new = remove_node_without_label(WP_G)
    # print('---------------------------------------')
    # temp_list = []
    # for node in WP_G_new:
    #     temp = node.split()
    #     if len(temp) == 4:
    #         temp_list.append(node.split())
    # # 记录cover值
    # if record is True:
    #     record_dic[node_codeword] = temp_list
    # # 与非cover做对比
    # else:
    #     compare_list = record_dic[node_codeword]
    #     compare_result = compare_node(compare_list, temp_list)
    #     print(compare_list)
    #     print('----------------')
    #     print(temp_list)
    #     print('----------------')
    #     print(compare_result)

# Load data
data_root = 'D:\\博士\\贝叶斯网络\\data\\'
subroots = ['cover', 'stego0.5', 'stego1.0']
labels = ['0.0', '0.5', '1.0']
roots_and_labels = [('cover', 0.0), ('stego0.5', 0.5), ('stego1.0', 1.0)]
# roots_and_labels = [('cover', 0.0), ('stego1.0', 0.5)]
record = True

for root_and_label in roots_and_labels:
    (subroot, label) = root_and_label
    # print('____________________________________________')
    # print(label)
    data_path = data_root + subroot + '\\data.csv'
    data = pd.read_csv(data_path)
    # Choose node and output corresponding directed acyclic graph.
    # codewords = ['L0', 'L1', 'L2', 'L3', 'P1', 'P0', 'C1', 'S1', 'GA1', 'GB1', 'P2', 'C2', 'S2', 'GA2', 'GB2']
    codewords = ['L0', 'L1', 'L2', 'L3']
    output_path_root = 'D:\\博士\\贝叶斯网络\\out\\' + subroot + '\\'
    for codeword in codewords:
        output_path = output_path_root + codeword + '\\'
        create_naive_bayes_figure(codeword, output_path, str(label), record)
    record = False
