# tUMPC

The code and data for tUMPC, introduced in "(Morphological Processing of Low-Resource Languages:
Where We Are and What’s Next)[https://aclanthology.org/2022.findings-acl.80.pdf]". The code here assumes a paradigm induction system has been run (we supply the induced paradigms we use int he paper in `data/clustered`), and uses those along with a corpus to generate slot-aligned inflection training data, and slot-tagging data.

## Usage

Run the "pos-based" system, for which we report the main results in the paper: `bash run_pos_based.sh`

## Dependencies

Depends on the [foma C library](https://github.com/mhulden/foma/tree/master/foma). 

## Data

### Train data
Train data is just input corpora with no information besides tokenization.

- `train/bible` raw bible corpora from JHU bible corpus [1].
- `train/child` child-directed book corpora, originally in English and automatically translated in each language.

## Cluster data
The induced paradigms from running various paradigm clustering systems

- `clustered/2021baseline` Paradigm clusters from SIGMORPHON 2021 shared task baseline [2]
- `clustered/McCurdyEtAl` Paradigm clusters from winning system of SIGMORPHON 2021 shared task [3]
- `clustered/XuEtAl` Paradigm clusters running the Xu et al segmentation algorithm, a byproduct of which is paradigms. [4]

## References
[1] Arya D. McCarthy, Rachel Wicks, Dylan Lewis, Aaron Mueller, Winston Wu, Oliver Adams, Garrett Nicolai, Matt Post, and David Yarowsky. 2020b. TheJohns Hopkins University Bible corpus:1600+tongues for typological exploration. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 2884–2892, Marseille, France. European Language Resources Association.

[2] Adam Wiemerslage, Arya D. McCarthy, AlexanderErdmann, Garrett Nicolai, Manex Agirrezabal, Miikka Silfverberg, Mans Hulden, and Katharina Kann. 2021. Findings of the SIGMORPHON 2021 shared task on unsupervised morphological paradigm clustering. In Proceedings of the 18th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 72–81, Online. Association for Computational Linguistics.

[3] Kate McCurdy, Sharon Goldwater, and Adam Lopez. 2021. Adaptor Grammars for unsupervised paradigm clustering. In Proceedings of the 18th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, pages 82–89, Online. Association for Computational Linguistics.

[4] Hongzhi Xu, Jordan Kodner, Mitchell Marcus, and Charles Yang. 2020. Modeling morphological typology for unsupervised learning of language morphology. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6672–6681, Online. Association for Computational Linguistics.

## Citation

```
@inproceedings{wiemerslageetal-2022-morphological,
    title = "Morphological Processing of Low-Resource Languages: Where We Are and What{'}s Next",
    author = "Wiemerslage, Adam  and
      Silfverberg, Miikka  and
      Yang, Changbing  and
      McCarthy, Arya  and
      Nicolai, Garrett  and
      Colunga, Eliana  and
      Kann, Katharina",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.80",
    doi = "10.18653/v1/2022.findings-acl.80",
    pages = "988--1007",
}
```
