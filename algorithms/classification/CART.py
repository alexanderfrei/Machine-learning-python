# TODO descriptions

my_data=[
    ['slashdot','USA','yes',18,'None'],
    ['google','France','yes',23,'Premium'],
    ['digg','USA','yes',24,'Basic'],
    ['kiwitobes','France','yes',23,'Basic'],
    ['google','UK','no',21,'Premium'],
    ['(direct)','New Zealand','no',12,'None'],
    ['(direct)','UK','no',21,'Basic'],
    ['google','USA','no',24,'Premium'],
    ['slashdot','France','yes',19,'None'],
    ['digg','USA','no',18,'None'],
    ['google','UK','no',18,'None'],
    ['kiwitobes','UK','no',19,'None'],
    ['digg','New Zealand','yes',12,'Basic'],
    ['slashdot','UK','no',21,'None'],
    ['google','UK','yes',18,'Basic'],
    ['kiwitobes','France','yes',19,'Basic']]

class decision_node:

    def __init__(self, col=-1, value=None, results=None,tb=None,fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def divide_set(rows, column, value):
    split_function = None
    # check if numeric column or categorical
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return set1, set2


def unique_counts(rows):
    # results counter
    results={}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def gini(rows):
    total = len(rows)
    counts = unique_counts(rows)
    imp = 0
    print(counts)
    for k1 in counts:
        p1 = counts[k1] / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = counts[k2] / total
            imp += p1 * p2
    return imp


def entropy(rows):
    from math import log
    results = unique_counts(rows)
    ent = 0
    for r in results:
        p = results[r] / len(rows)
        ent -= p*log(p, 2)
    return ent


def build_tree(rows, score_f=entropy):

    if len(rows) == 0:
        return decision_node()
    current_score = score_f(rows)

    best_gain = 0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1

    # information gain
    for col in range(0, column_count):
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1

        for value in column_values:
            (set1, set2) = divide_set(rows, col, value)
            p = len(set1) / len(rows)
            gain = current_score - p * score_f(set1) - (1 - p) * score_f(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # build tree
    if best_gain > 0:
        true_branch = build_tree(best_sets[0])
        false_branch = build_tree(best_sets[1])
        return decision_node(col=best_criteria[0], value=best_criteria[1],
                             tb=true_branch, fb=false_branch)
    else:
        return decision_node(results=unique_counts(rows))


def print_tree(tree, indent=''):
    # leaves
    if tree.results:
        print(tree.results)
    else:
        print("{}{}{}{}".format(tree.col,':',tree.value,'?'))
        print(indent + 'T->', end='')
        print_tree(tree.tb, indent+' ')
        print(indent + 'F->', end='')
        print_tree(tree.fb, indent+' ')

def classify(obs, tree):
    if tree.results:
        return tree.results
    else:
        v = obs[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(obs, branch)


def md_classify(obs, tree):
    if tree.results:
        return tree.results
    else:
        v = obs[tree.col]
        if not v:
            tr, fr = md_classify(obs, tree.tb), md_classify(obs, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in tr.items():
                result[k] = v * tw
            for k, v in fr.items():
                result[k] = v * fw
            return result
        else:
            v = obs[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return md_classify(obs, branch)


def prune(tree, mingain):

    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)

    if tree.tb.results and tree.fb.results:
        # check leaves for union necessity
        tb, fb = [], []
        for v,c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        # estimate entropy delta of general node and mean of children nodes
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb))/2
        if delta < mingain:
            tree.tb, tree.fb = None, None
            tree.results = unique_counts(tb + fb)



def tree_var(rows):
    # for float and int features
    if len(rows) == 0:
        return 0
    data = [row[len(row) - 1] for row in rows]
    mean = sum(data) / len(data)
    var = sum([(d - mean) ** 2 for d in data]) / len(data)
    return var


tree = build_tree(my_data)
print(md_classify(['google',None,'yes',None],tree))
print(md_classify(['google','France',None,None],tree))

# print(classify(['(direct)', 'USA', 'yes', 5], tree))
# prune(tree, 1)
# print_tree(tree)

