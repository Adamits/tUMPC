def get_keep():
    keep = []

    with open("sentences.txt", "r") as f:
        for line in f:
            sent, idx = line.strip().split("\t")
            w = sent.split()[int(idx)]
            if w != "—":
                keep.append(int(idx))

    print(len(keep))
    return keep

def filter(fn, keep):
    with open(fn + ".fixed", "w") as o:
        with open(fn, "r") as f:
            for i, line in enumerate(f):
                if i in keep:
                    o.write(line)

K = get_keep()
filter("metadata.tsv", K)
filter("sentences.txt", K)

paradigms = []
P = []
with open("paradigms.txt", "r") as f:
    for line in f:
        if len(line.strip()) < 1:
            paradigms.append(P)
            P = []
        else:
            P.append(line)

print(len(paradigms))
with open("paradigms.txt.fixed", "w") as o:
    for i, p in enumerate(paradigms):
        if i in K:
            o.write("".join(p))
            o.write("\n")