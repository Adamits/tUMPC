from asyncio import constants
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import Counter

import foma

from constants import *

# for reproducibility
seed(0)


def relcs(forms: str) -> List:
    """Get the longest common subsequence for a set of forms"""
    regexes = ["{%s}" % f for f in forms]
    fsts = [foma.FST(re) for re in regexes]

    opt_del_fst = foma.FST("[?|?:0]*")

    common_string_fst = foma.FST("?*")
    for fst in fsts:
        common_string_fst = common_string_fst & fst.compose(opt_del_fst).lower()

    common_strings = sorted([w for w in common_string_fst.words()],key=lambda x: len(x), reverse=True)
    return common_strings[0]


def alignVariables(sqs: List[str], variables: List[str]) -> List[str]:
    """Align the seqs, replacing each `variable` string with an identifier of the form: Xi
    
    e.g. sqs:       ['ausländer', 'ausländische', 'ausländischen', 'ausländern'] 
         variables: ['ausländ', 'e']
    --> 'X0+X1+r', 'X0+isch+X1', 'X0+isch+X1+n', 'X0+X1+rn']"""
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


def greedyAlign(lcs: str, sqs: List[str]) -> Tuple[Dict[Tuple, str], List[str]]:
    """Make the abstract paradigms, replacing common substrings with variables.
    
    e.g. lcs: auslände 
         sqs: ['ausländer', 'ausländische', 'ausländischen', 'ausländern']
        --> ['X0+X1+r', 'X0+isch+X1', 'X0+isch+X1+n', 'X0+X1+rn']
    """
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


def abstractParadigm(sqs: List) -> Tuple[List[str], Dict[str, str], str]:
    """Produce abstract paradigms.
    
    Return: 
        aligned_forms: the List of abstract paradigm
        variables:     the Dict mapping a variable to the string it represents
        lcs:           the longest common subsequence string for the strings in sqs
    """
    lcs = relcs(sqs)
    variables, aligned_forms = greedyAlign(lcs,sqs)
    return aligned_forms, variables, lcs


def get_resampled_forms(p: Dict[str, List], suff_len: int) -> List[str]:
    """Update each abstract form by increasing the size of the suffix"""
    forms = []
    for a, f in zip(p["abs_par"],p["tokens"]):
        if a.count("+") > 1:
            return []
        _, suff = a.split("+") if "+" in a else ("X0","")
        new_suff = f[len(f)-len(suff)-suff_len:]
        if new_suff == f:
            return []

        print("X0+%s" % new_suff)
        forms.append("X0+%s" % new_suff)

    return forms
    

def resample_abs_par(data: Dict[str, List], form_counts: Dict[str, int]):
    """Check each abstract paradigm for a resampling that results in a higher frequency of abstract forms.
    
    Return the abstract paradigm that maximizes the sum of its form frequencies."""
    for p in data:
        # Not a singleton paradigm
        if len(p["abs_par"]) > 1:
            suff_len = 1
            # sum of abstract form frequencies in the paradigm not counting the base form X0
            max_score = sum([form_counts[f] for f in p["abs_par"] if f in form_counts if f != "X0"])
            max_forms = p["abs_par"]

            # Find the abstract paradigm that maximizes frequency
            while 1:
                forms = get_resampled_forms(p,suff_len)
                # Loop until we have tried every possible suffix.
                if forms == []:
                    break
                score = sum([form_counts[f] for f in forms if f in form_counts if f != "X0"])
                if score > max_score:
                    max_forms = forms
                    max_score = score
                suff_len+=1

            p["abs_par"] = max_forms

    return data


def abstract_and_filter(data: Dict[str, List]) -> Tuple(Dict[str, List], Dict[str, int]):
    """Produce abstract paradigms from the paradigms
    
    Filter out infrequent abstract forms"""
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

    # Resample to maximize form frequency
    data = resample_abs_par(data,form_counts)

    form_counts = Counter()
    for p in data:
        form_counts.update(p["abs_par"])

    # Keep only abstract forms that occur more than MIN_ABS_COUNT times across all paradigms
    forms = {f for f,c in form_counts.items() if c >= MIN_ABS_COUNT}

    # Filter to the abs forms and corresponding tokens that are frequent.
    for p in data:
        p["filtered_tokens"] = []
        p["filtered_abs_par"] = []
        for f, a in zip(p["tokens"], p["abs_par"]):
            if a in forms:
                p["filtered_tokens"].append(f)
                p["filtered_abs_par"].append(a)

    # No singletons
    data = [p for p in data if len(p["filtered_tokens"]) > 1]

    print("Got %u paradigms with at least two forms" % len(data))
    print("Got %u distinct abstract forms with at least %u occurrences" % (len(forms),MIN_ABS_COUNT))

    return data, forms