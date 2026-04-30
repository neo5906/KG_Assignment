# src/kg_visualizer.py
import json
from pyvis.network import Network

# ========== 配置 ==========
TRIPLES_FILE = 'data/output/turing_triples_cs_finetuned.json'
OUTPUT_HTML = 'data/output/knowledge_graph.html'
# ==========================

def load_triples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_graph(triples):
    nodes_set = set()
    edges = []
    for item in triples:
        s, p, o = item['subject'], item['predicate'], item['object']
        nodes_set.add(s)
        nodes_set.add(o)
        edges.append((s, o, p))

    relation_colors = {}
    palette = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
        '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
    ]
    color_idx = 0
    for s, t, r in edges:
        if r not in relation_colors:
            relation_colors[r] = palette[color_idx % len(palette)]
            color_idx += 1

    links = []
    for s, t, r in edges:
        links.append({
            'source': s,
            'target': t,
            'label': r,
            'color': relation_colors[r],
            'width': 1.5  # 边变细一点
        })
    return nodes_set, links, relation_colors

def render_graph(nodes, links, relation_colors, output_path):
    net = Network(height='900px', width='100%', directed=True, notebook=False)

    for name in nodes:
        net.add_node(name, label=name, size=15, font={'size': 12})

    for link in links:
        net.add_edge(
            link['source'], link['target'],
            title=link['label'],
            label=link['label'],
            color=link['color'],
            width=link['width'],
            physics=True
        )

    options_json = """
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.2,
          "springLength": 250,
          "springConstant": 0.01,
          "damping": 0.12
        },
        "minVelocity": 0.5,
        "solver": "barnesHut",
        "timestep": 0.5,
        "stabilization": {
          "iterations": 200,
          "updateInterval": 25
        }
      },
      "nodes": {
        "scaling": {
          "min": 15,
          "max": 30
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        },
        "font": {
          "size": 8,
          "align": "middle"
        }
      }
    }
    """
    net.set_options(options_json)
    net.write_html(output_path)
    print(f'知识图谱已生成：{output_path}')

def main():
    triples = load_triples(TRIPLES_FILE)
    print(f'加载 {len(triples)} 个三元组')
    nodes, links, relation_colors = build_graph(triples)
    print(f'节点数：{len(nodes)}，边数：{len(links)}，关系类型：{len(relation_colors)}')
    render_graph(nodes, links, relation_colors, OUTPUT_HTML)

if __name__ == '__main__':
    main()