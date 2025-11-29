import json
from tqdm import tqdm
import numpy as np

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def list_equal(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        check = False
        for j in range(len(l1)):
            if l1[i] == l2[j]:
                check = True
        if not check:
            return False
    return True

def check_add(arr, item):
    add = True
    for i in arr:
        if len(i) != len(item):
            continue
        if list_equal(i, item):
            add = False
            break
    if add:
        arr.append(item)


def idx_of(arr, item):
    for index, i in enumerate(arr):
        if len(i) != len(item):
            continue
        if list_equal(i, item):
            return index
    raise Exception("No way.")


def sort_vertex(vertexs):
    temp = list()

    for v in vertexs:
        t = sorted(v, key=lambda x: (x['sent_id'], x['pos'][0]))
        sel = t[0]
        temp.append(sel)

    sorted_indices = sorted(
        range(len(temp)),
        key=lambda i: (temp[i]['sent_id'], temp[i]['pos'][0])
    )

    data_sorted = [vertexs[i] for i in sorted_indices]
    return data_sorted


def transform(file_in, file_out, dataset):
    output = []
    pmids = set()

    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)): #each document
            per_doc = {}
            line = line.rstrip().split('\t')
            pmid = line[0] # Doc ID

            if pmid in pmids:
                continue
            pmids.add(pmid)
            text = line[1]
            prs = chunks(line[2:], 17)

            vertexs = list() # each entity
            labels = []
            title = pmid
            sents = [t.split(' ') for t in text.split('|')]
            len_sents = [len(i) for i in sents]

            vertex1_cache = dict()
            vertex2_cache = dict()

            for idx, p in enumerate(prs):  # each triplet
                if p[0] == 'not_include':
                    continue
                vertex1 = list() # first entity in the triplet, list of mentions
                mention_names1 = p[6].split('|')
                if dataset == 'cdr':
                    type1 = "CHEM" if p[7] == 'Chemical' else "DISE" if p[7] == 'Disease' else None
                elif dataset =='gda':
                    type1 = "GENE" if p[7] == 'Gene' else "DISE" if p[7] == 'Disease' else None
                assert type1 is not None
                sent_ids1 = p[10].split(':')
                positions_start1 = p[8].split(':')
                positions_end1 = p[9].split(':')
                for i in range(len(mention_names1)):
                    p1 = int(positions_start1[i]) - np.sum(len_sents[:int(sent_ids1[i])])
                    p2 = int(positions_end1[i]) - np.sum(len_sents[:int(sent_ids1[i])])
                    vertex1.append({'pos': [int(p1), int(p2)],
                                    'type': type1,
                                    'sent_id': int(sent_ids1[i]),
                                    'name': mention_names1[i]
                                    })

                vertex2 = list() # first entity in the triplet, list of mentions
                mention_names2 = p[12].split('|')
                if dataset == 'cdr':
                    type2 = "CHEM" if p[13] == 'Chemical' else "DISE" if p[13] == 'Disease' else None
                elif dataset == 'gda':
                    type2 = "GENE" if p[13] == 'Gene' else "DISE" if p[13] == 'Disease' else None
                assert type2 is not None
                sent_ids2 = p[16].split(':')
                positions_start2 = p[14].split(':')
                positions_end2 = p[15].split(':')
                for i in range(len(mention_names2)):
                    p1 = int(positions_start2[i]) - np.sum(len_sents[:int(sent_ids2[i])])
                    p2 = int(positions_end2[i]) - np.sum(len_sents[:int(sent_ids2[i])])
                    vertex2.append({'pos': [int(p1), int(p2)],
                                    'type': type2,
                                    'sent_id': int(sent_ids2[i]),
                                    'name': mention_names2[i]
                                    })
                check_add(vertexs, vertex1)
                check_add(vertexs, vertex2)

                vertex1_cache[idx] = vertex1
                vertex2_cache[idx] = vertex2

            #sort
            vertexs = sort_vertex(vertexs)

            for idx, p in enumerate(prs):
                if p[0] == 'not_include':
                    continue
                vertex1 = vertex1_cache[idx]
                vertex2 = vertex2_cache[idx]
                id1 = idx_of(vertexs, vertex1)
                id2 = idx_of(vertexs, vertex2)
                r = p[0].split(':')[1]
                if dataset == 'cdr':
                    assert (r == 'CID') or (r == 'NR')
                elif dataset == 'gda':
                    assert (r == 'GDA') or (r == 'NR')
                h = id1
                t = id2

                labels.append({'r': r, 'h': h, 't': t, 'evidence': [None]})


            per_doc['vertexSet'] = vertexs
            per_doc['labels'] = labels
            per_doc['title'] = title
            per_doc['sents'] = sents
            output.append(per_doc)

    with open(file_out, "w") as final:
        json.dump(output, final)

import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def heat_map(data, out_file):
    max_l = -1
    for h, t in data.keys():
        max_l = max(max_l, h, t)

    mt = np.zeros((max_l + 1, max_l + 1), dtype=np.int64)

    for key in data.keys():
        tmp = data[key]
        h, t = key
        mt[h, t] = tmp

    plt.figure(figsize=(8, 6))

    annot = (mt.size <= 1000)
    if not annot:
        mt = mt[:10, :10]
    annot = True

    sns.heatmap(
        mt,
        cmap="viridis",
        annot=annot,
        fmt="d" if annot else "",
    )

    plt.title(out_file)
    plt.tight_layout()

    plt.savefig(out_file + ".jpg", dpi=300)

    plt.close()

def extract_data(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    res = dict()
    count = 0
    for idx, d in enumerate(data):
        labels = d['labels']
        for l in labels:
            if l['r'] != 'GDA':
                continue
            h = l['h']
            t = l['t']
            ht = (h, t)
            if ht in res:
                res[ht] += 1
            else:
                res[ht] = 1
    return res


if __name__ == '__main__':
    # transform('data/cdr/train_filter.data', 'train.json', 'cdr')
    # transform('data/cdr/dev_filter.data', 'dev.json', 'cdr')
    # transform('data/cdr/test_filter.data', 'test.json', 'cdr')


    transform('data/gda/train.data', 'train.json', 'gda')
    transform('data/gda/dev.data', 'dev.json', 'gda')
    transform('data/gda/test.data', 'test.json', 'gda')

    train_stat = extract_data('train.json')
    heat_map(train_stat, "train")

    dev_stat = extract_data('dev.json')
    heat_map(dev_stat, "dev")

    test_stat = extract_data('test.json')
    heat_map(test_stat, "test")
