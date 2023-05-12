import click
from collections import Counter, defaultdict
import numpy as np
from math import log
from random import randint, sample, seed, shuffle
from sklearn.cluster import AgglomerativeClustering
import fasttext
from tqdm import tqdm
import foma

# for reproducibility
seed(0)

MIN_ABS_COUNT=50
POS_N=3
CONTEXT_N=5
CONTEXT_TOK_TH=50
CLUSTER_TH=0.01
MAX_POS_EX=1500
ALPHA=1
MERGE_TH=0.9

def relcs(forms):
    regexes = ["{%s}" % f for f in forms]
    fsts = [foma.FST(re) for re in regexes]

    opt_del_fst = foma.FST("[?|?:0]*")

    common_string_fst = foma.FST("?*")
    for fst in fsts:
        common_string_fst = common_string_fst & fst.compose(opt_del_fst).lower()

    common_strings = sorted([w for w in common_string_fst.words()],key=lambda x: len(x), reverse=True)
    return common_strings[0]

def alignVariables(sqs, variables):
    results = []
    for sq in sqs:
        res = ""
        curr = 0
        for i,v in enumerate(variables):
            match = sq.find(v,curr)
            if i > 0:
                res += "+"
            if curr < match:
                res += sq[curr:match] + "+"
            res += ("X%u" % i)
            
            curr = match + len(v)
        if curr < len(sq):
            res += "+" + sq[curr:]
        
        results.append(res)
    return results

def greedyAlign(lcs, sqs):
    indices = [0 for sq in sqs]
    variables = [""]
    for c in lcs:
        split = False
        for i in range(len(indices)):
            while indices[i] < len(sqs[i]) and sqs[i][indices[i]] != c:
                split = True
                indices[i] += 1
            indices[i] += 1
        if split:
            variables += [""]
        variables[-1] += c
    variables = [v for v in variables if v]
    
    return {("X%u" % i):v for i,v in enumerate(variables)}, alignVariables(sqs,variables)

def abstractParadigm(sqs):
    lcs = relcs(sqs)
    variables, aligned_forms = greedyAlign(lcs,sqs)
    return aligned_forms, variables, lcs

def read(fn):
    print("Reading data")
    # data = [p.lower().split("\n") for p in open(fn).read().split("\n\n") if p]
    data = [p.split("\n") for p in open(fn).read().split("\n\n") if p]
    paradigms = []
    for p in data:
        paradigm = {}
        paradigm["tokens"] = p
        paradigms.append(paradigm)
    return paradigms

def get_resampled_forms(p,suff_len):
    forms = []
    for a, f in zip(p["abs_par"],p["tokens"]):
        if a.count("+") > 1:
            return []
        _, suff = a.split("+") if "+" in a else ("X0","")
        new_suff = f[len(f)-len(suff)-suff_len:]
        if new_suff == f:
            return []
        forms.append("X0+%s" % new_suff)
    return forms
    
def resample_abs_par(data,form_counts):
    for p in data:
        if len(p["abs_par"]) > 1:
            suff_len = 1
            max_score = sum([form_counts[f] for f in p["abs_par"] if f in form_counts if f != "X0"])
            max_forms = p["abs_par"]
            while 1:
                forms = get_resampled_forms(p,suff_len)
                if forms == []:
                    break
                score = sum([form_counts[f] for f in forms if f in form_counts if f != "X0"])
                if score > max_score:
                    max_forms = forms
                    max_score = score
                suff_len +=1
            p["abs_par"] = max_forms
    return data

def abstract_and_filter(data):
    print("Generating abstract paradigms (can take a while)")
    form_counts = Counter()
    for p in tqdm(data):
        # Speeds things along a bit for very long words
        if len(p["tokens"]) == 1:
            p["abs_par"] = ["X0"]
        else:
            aligned_forms, variables, _ = abstractParadigm(p["tokens"])
            p["abs_par"] = aligned_forms
        form_counts.update(p["abs_par"])
    data = resample_abs_par(data,form_counts)

    form_counts = Counter()
    for p in data:
        form_counts.update(p["abs_par"])
    forms = {f for f,c in form_counts.items() if c >= MIN_ABS_COUNT}

    for p in data:
        p["filtered_tokens"] = []
        p["filtered_abs_par"] = []
        for f, a in zip(p["tokens"], p["abs_par"]):
            if a in forms:
                p["filtered_tokens"].append(f)
                p["filtered_abs_par"].append(a)

    data = [p for p in data if len(p["filtered_tokens"]) > 1]

    print("Got %u paradigms with at least two forms" % len(data))
    print("Got %u distinct abstract forms with at least %u occurrences" % (len(forms),MIN_ABS_COUNT))

    return data, forms

def e_step(paradigms, abs_forms):
    pos_distr = {pos:ALPHA for pos in range(POS_N)}
    abs_form_distr = {pos:{f:ALPHA for f in abs_forms} for pos in range(POS_N)}
    
    for p in paradigms:
        for pos in range(POS_N):
            pos_distr[pos] += float(p["distr"][pos])
            for f in p["filtered_abs_par"]:
                if f in abs_form_distr[pos]:
                    abs_form_distr[pos][f] += float(p["distr"][pos])

    tot = sum(pos_distr.values())
    for pos in range(POS_N):
        pos_distr[pos] /= tot
        pos_tot = sum(abs_form_distr[pos].values())
        for abs_form in abs_form_distr[pos]:
            if abs_form in abs_form_distr[pos]:
                abs_form_distr[pos][abs_form] /= pos_tot
    return pos_distr, abs_form_distr

def m_step(paradigms, pos_distr, abs_form_distr):
    for p in paradigms:
        new_distr = np.array([0.0 for i in range(POS_N)])
        for pos in range(POS_N):
            prob = pos_distr[pos]
            for abs_form in p["filtered_abs_par"]:
                if abs_form == "X0":
                    continue
                if abs_form in abs_form_distr[pos]:
                    prob *= abs_form_distr[pos][abs_form]
            new_distr[pos] = prob
        if new_distr.sum() > 0:
            p["distr"] = new_distr / new_distr.sum()
        else:
            p["distr"] = new_distr

def get_distr():
    distr = None
    distr = np.zeros((POS_N,1))
    distr[randint(0,POS_N-1)] = 1
    return (distr / distr.sum())

def identify_pos(data,abs_forms):
    print("Identify POS (running EM)")
    for p in data:
        p["distr"] = get_distr()

    pos_distr, abs_form_distr = None, None
    for i in tqdm(range(1000)):
        pos_distr, abs_form_distr = e_step(data, abs_forms)
        m_step(data, pos_distr, abs_form_distr)
    print("")

    poses = [[] for i in range(POS_N)]

    for p in data:
        pos = poses[p["distr"].argmax()]
        pos.append(p)

    for i,pos in enumerate(poses):
        print("POS",i,":",len(pos),"lexemes")
        print("Sample paradigms:")
        for p in sample(pos,min(10,len(pos))):
            print(" ".join(p["filtered_tokens"])," ".join(p["filtered_abs_par"]))
        print("Most frequent abstract forms:")
        print(Counter([f for p in pos for f in p["filtered_abs_par"]]).most_common(5))
        print()                
    return poses

def sim(a1,a2,embedding):
    return np.dot(embedding[a1],embedding[a2])/(np.linalg.norm(embedding[a1])*np.linalg.norm(embedding[a2]))

def align_distr(d1,d2,embedding):
    similarities = [(sim(a1,a2,embedding), (a1,a2)) for (f1,a1) in d1 for (f2,a2) in d2
                    if a1 in embedding and a2 in embedding]
    similarities.sort(reverse=True)
    pairs = []
    found1 = set()
    found2 = set()
    d1 = {a:f for f,a in d1}
    
    for score, (a1,a2) in similarities:
        if not a1 in found1:
            found1.add(a1)
            pairs.append((score,(a1,a2)))

    return sum([score for score, _ in pairs])/len(pairs)
    
def combine_pos(data,embedding):
    print("Merging POS classes")
    counters = []
    for pos in data:
        counters.append(Counter([a for p in pos for a in p["abs_par"]]))
    distrs = []    
    for counter in counters:
        tot = sum(counter.values())
        distrs.append([(c/tot,a) for a,c in counter.most_common(4)])

    print()
    top_pair = None
    top_score = -1
    for i1,d1 in enumerate(distrs):
        for i2,d2 in enumerate(distrs):
            if i1 < i2:
                score = align_distr(d1,d2,embedding)*align_distr(d2,d1,embedding)
                if score > top_score:
                    top_score = score
                    top_pair = (i1,i2)
    if top_score > MERGE_TH:
        print("Similarity %f > threshold %f. Merging POS %u and POS %u" % (top_score,MERGE_TH,top_pair[0],top_pair[1]))
        data[top_pair[0]] += data[top_pair[1]]
        data.pop(top_pair[1])
        return True
    else:
        print("Similarity of POS %u and %u is %f <= threshold %f. Nothing to merge" % (top_pair[0],top_pair[1],top_score,MERGE_TH))
        print("Nothing to merge")
    return False

def detect_slots(data, embedding):
    print("Merging paradigms slots.")    
    for pos_i, pos in enumerate(data):
        if len(pos) == 0:
            continue
        pair_counts = Counter()
        form_counts = Counter()

        for p in pos:
            pair_counts.update([(f1,f2) for f1 in p["filtered_abs_par"] for f2 in p["filtered_abs_par"]])
            form_counts.update(p["filtered_abs_par"])
        form_counts_total = sum(form_counts.values())
        forms = set()
        for p in pos:
            abs_par = []
            tokens = []
            for f, a in zip(p["filtered_tokens"],p["filtered_abs_par"]):
                if form_counts[a]/form_counts_total < CLUSTER_TH:
                    continue
                abs_par.append(a)
                tokens.append(f)
                forms.add(a)
            p["filtered_tokens"] = tokens
            p["filtered_abs_par"] = abs_par
                
        abs_pair_count = len(form_counts)**2
        pair_count_total = sum(pair_counts.values())
        joint_prob_pair = lambda p: (pair_counts[p] + ALPHA) / (pair_count_total + abs_pair_count * ALPHA) 
        cond_prob_pair = lambda p: (pair_counts[p] + ALPHA) / (form_counts[p[1]] + len(form_counts) * ALPHA) 
        emb_sim = lambda p: np.dot(embedding[p[0]],embedding[p[1]])
            
        array = []
#        forms = list(form_counts.keys())
        print("Got %s distinct abstract forms" % len(forms))
        for f1 in forms:
            array.append([])
            for f2 in forms:
                d = 1 - emb_sim((f1,f2)) * (1 - pair_counts[(f1,f2)]/(form_counts[f1] + form_counts[f2] - pair_counts[(f1,f2)]))
                if f1 == f2:
                    array[-1].append(0)
                else:
                    array[-1].append(d)

        clustering = AgglomerativeClustering(affinity="precomputed",linkage="average",n_clusters=None,distance_threshold=0.15).fit(array)

        slots = {f:l for f,l in zip(forms,clustering.labels_)}
        print(slots)
        for paradigm in pos:
            paradigm["labels"] = []
            for form in paradigm["filtered_abs_par"]:
                if form in slots:
                    paradigm["labels"].append(slots[form])
                else:
                    paradigm["labels"].append(-1)

        print("POS %u:" % pos_i)
        for i in range(max(clustering.labels_)+1):
            print(" Cluster %u:" % i," ".join(["%s (%.2f%%)" % (f,100*form_counts[f]/form_counts_total) for f in forms if slots[f] == i]))
        print()
    return data

def train_embedding(data,corpus):
    print("Training embeddings")
    afdict = {}
    form_counts = Counter()
    form_pair_counts = Counter()
    embeddings = fasttext.train_unsupervised(corpus, ws=1,epoch=100)
    
    abstract_embeddings = defaultdict(list)

    for pos in data:
        for p in pos:
            for f, af in zip(p["filtered_tokens"],p["filtered_abs_par"]):
                if f in embeddings:
                    abstract_embeddings[af].append(embeddings.get_word_vector(f))
                # elif f.lower() in embeddings:
                #     abstract_embeddings[af].append(embeddings.get_word_vector(f.lower()))
                elif f in embeddings:
                    abstract_embeddings[af].append(embeddings.get_word_vector(f))
            for af1 in p["filtered_abs_par"]:
                form_counts.update([af1])
                for af2 in p["filtered_abs_par"]:
                    form_pair_counts.update([(af1,af2)])
                    
    for af1 in form_counts:
        for af2 in form_counts:
            form_pair_counts[(af1,af2)] += 1
    mutual_information = defaultdict(dict)
    total_count = sum(form_counts.values())
    total_pair_count = sum(form_pair_counts.values())

    for af1 in form_counts:
        for af2 in form_counts:
            p_xy = form_pair_counts[(af1,af2)]/total_pair_count
            p_x = form_counts[af1]/total_count
            p_y = form_counts[af2]/total_count
            mutual_information[af1][af2] = -log(p_xy/(p_x*p_y))
        
    for af in abstract_embeddings:
        abstract_embeddings[af] = np.add.reduce(abstract_embeddings[af])/len(abstract_embeddings[af])
        abstract_embeddings[af] = abstract_embeddings[af]/np.linalg.norm(abstract_embeddings[af])

    return abstract_embeddings

def write_inflection_data(data,output):
    print("Writing inflection data")
    output = {(split,dire):open("%s.infl-%s.%s" % (output, split, dire),"w")
              for split in "train valid".split(" ")
              for dire in "src tgt".split(" ")}
    
    for i, pos in enumerate(data):
        for j, p in enumerate(pos):
            split = "valid" if j % 10 == 0 else "train"
            for f1, l1 in zip(p["filtered_tokens"],p["labels"]):
                for f2, l2 in zip(p["filtered_tokens"],p["labels"]):
                    print("%s POS=%u SRC_LABEL=%u TGT_LABEL=%u" % (" ".join(f1),i,l1,l2),file=output[(split,"src")])
                    print(" ".join(f2),file=output[(split,"tgt")])
            
def write_contexts(data,corpus,output):
    print("Writing POS and slot prediction data")
    output = {(split,dire):open("%s.pos-%s.%s" % (output, split, dire),"w")
              for split in "train valid".split(" ")
              for dire in "src tgt".split(" ")}
    
    corpus = [l.strip().split(" ") for l in open(corpus).read().split("\n") if l]
    index = defaultdict(list)
    for i,line in enumerate(corpus):
        for j,tok in enumerate(line):
            index[tok].append({"line_id":i,"tok_id":j})
    
    for i, pos in enumerate(data):
        for j, p in enumerate(pos):
            split = "valid" if j % 10 == 0 else "train"
            for form, label in zip(p["filtered_tokens"],p["labels"]):
                contexts = index[form] if form in index else index[form.lower()] if form.lower() in index else []
                context_ids = sample(contexts, min(CONTEXT_N,len(contexts)))
                for c in context_ids:
                    p_tok = corpus[c["line_id"]][c["tok_id"]-1] if c["tok_id"] > 0 else "<"
                    n_tok = corpus[c["line_id"]][c["tok_id"]+1] if c["tok_id"] + 1 < len(corpus[c["line_id"]]) else ">"
                    if p_tok != "<" and len(index[p_tok]) < CONTEXT_TOK_TH:
                        p_tok = "R"
                    if n_tok != ">" and len(index[n_tok]) < CONTEXT_TOK_TH:
                        n_tok = "R"
                    print("%s # %s # %s" % (" ".join(p_tok)," ".join(form)," ".join(n_tok)), file=output[(split,"src")])
                    print("POS=%u SRC_LABEL=%u" % (i,label),file=output[(split,"tgt")])

@click.command()
@click.option("--clusters",required=True)
@click.option("--corpus",required=True)
@click.option("--output",required=True)
def main(clusters,corpus,output):
    data = read(clusters)
    data, forms = abstract_and_filter(data)
    data = identify_pos(data,forms)

    abs_emb = train_embedding(data,corpus)
    while len(data) > 1 and combine_pos(data,abs_emb):
        pass
    data = detect_slots(data,abs_emb)

    for pos in data:
        shuffle(pos)
    write_inflection_data(data,output)
    write_contexts(data,corpus,output)
    
if __name__=="__main__":
    main()
