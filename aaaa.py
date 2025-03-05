import pandas as pd
import dill
import networkx as nx


# 步骤 1: 读取 Excel 文件
def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df


# 步骤 2: 构建图的节点和边
def build_graph(df):
    G = nx.Graph()

    # 根据指定的列名构建节点和边
    for index, row in df.iterrows():
        mirna = row['miRNA']
        target_gene = row['Target Gene']
        experiment = row['Experiments']
        reference = row['References (PMID)']

        # 添加节点
        G.add_node(mirna)
        G.add_node(target_gene)

        # 添加边，附带参考文献
        G.add_edge(mirna, target_gene, reference=reference)

    return G, df['Experiments']  # 返回图和实验文本


# 步骤 3: 序列化图数据到 .pk 文件
def serialize_graph(G, file_name):
    with open(file_name, 'wb') as f:
        dill.dump(G, f)


# 步骤 4: 从 .pk 文件中反序列化数据并打印结果
def deserialize_graph(file_name):
    with open(file_name, 'rb') as f:
        G = dill.load(f)
    return G


# 步骤 5: 提取文本特征（示例函数）
def extract_text_features(experiment_texts):
    # 在这里添加文本特征提取的方法，例如使用 TF-IDF 或其他技术
    text_features = [text.split() for text in experiment_texts]  # 示例：简单分词
    return text_features


# 主程序
def main(file_path):
    df = read_excel(file_path)
    G, experiments = build_graph(df)

    # 保存图数据
    serialize_graph(G, 'graph_data.pk')

    # 从文件中加载图
    loaded_graph = deserialize_graph('graph_data.pk')

    # 打印图的节点和边
    print("Nodes:", loaded_graph.nodes())
    print("Edges:", loaded_graph.edges(data=True))

    # 提取文本特征
    features = extract_text_features(experiments)
    print("Extracted Text Features:", features)


if __name__ == "__main__":
    excel_file_path = 'new/datasets/miRTarBase_MTI.xlsx' # 更新此路径
    main(excel_file_path)
