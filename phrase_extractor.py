#!/usr/bin/env python
# encoding: utf8

import re, string

class PhraseExtractor():
    is_a_tags = ['amod', 'advmod', 'nmod:npmod', 'acl', 'dep', 'nummod', 'compound:prt', 'nsubj', 'csubj', 'agent', 'obl:agent', 'npadvmod']
    subj_tags = ['nsubj', 'csubj', 'agent', 'obl:agent', 'expl', 'nmod', 'case', 'neg']
    obj_tags = ['dobj', 'dative', 'iobj', 'pobj', 'pcomp', 'xcomp', 'nsubjpass', 'csubjpass', 'acl:relcl', 'oprd', 'conj', 'case', 'acomp', 'appos', 'compound'] # 'attr'
    noun_tags = ['PROPN', 'NOUN', 'NUM', 'ADJ', 'DET', 'CCONJ', 'PART']
    right_noun_tags = ['PART']
    wh_regex = [
        'who',
        'what',
        'where',
        'when',
        'why',
        'how many',
        'how',
        'show',
        'which',
        '(are|is) the(re)?',
        'is',
        'are',
        'does',
        'did',
        'do',
        'can',
        'need',
        'have',
        'has',
        'pull up',
        'list',
    ]
    wh_map = {
        'are there': 'be there',
        'is there': 'be there',
        'is the': 'be the',
        'does': 'do',
        'did': 'do',
        'is': 'be',
        'are': 'be',
        'have': 'has'
    }

    @classmethod
    def get_np_lefts(cls, token):
        '''
        Progressively builds a noun phrase from the left side of the given token's dependency tree.

        @param token: spaCy Token.
        @return: Noun chunks as a list.
        '''

        nc = [token]

        for left in list(token.lefts)[::-1]:
            if left.pos_ not in PhraseExtractor.noun_tags: return nc
            
            if len(list(left.lefts)):
                nc += PhraseExtractor.get_np_lefts(left)
            
            else:
                for right in list(left.rights):
                    if right.pos_ not in PhraseExtractor.right_noun_tags: continue

                    nc.append(right)

                nc.append(left)

        return nc
    
    @classmethod
    def get_noun_chunks(cls, doc):
        '''
        Extracts noun chunks from given spaCy Document.

        @param doc: spaCy Document.
        @return: tuple containing list of noun chunks and list of POS tags associated with each noun chunk.
        '''
        ncs = []
        ncs_pos = []

        for nc in doc.noun_chunks:
            if nc.root.dep_ in PhraseExtractor.obj_tags or nc.root.pos_ in ('NOUN', 'PROPN'):
                ncs.append(' '.join([w.lemma_ for w in nc]))
                ncs_pos.append(' '.join([n.pos_ for n in nc]))

            noun_phrase = True

            for n in nc:
                if n.pos_ not in ('NOUN', 'PROPN', 'CCONJ'):
                    noun_phrase = False

                if noun_phrase:
                    ncs.append(' '.join([w.lemma_ for w in nc]))
                    ncs_pos.append(' '.join([n.pos_ for n in nc]))

            for _, sent in enumerate(doc.sents):
                for word in sent:
                    if word.dep_ in PhraseExtractor.is_a_tags:
                        d = {}

                        if word in list(word.head.lefts) or word.head.pos_ in PhraseExtractor.noun_tags:
                            nc = PhraseExtractor.get_np_lefts(word)
                            nc = nc[::-1]
                            noun_p = ' '.join([w.lemma_ for w in nc])
                            noun_pos = ' '.join([n.pos_ for n in nc])

                            if noun_p not in ncs:
                                ncs.append(noun_p)
                                ncs_pos.append(noun_pos)

        return ncs, ncs_pos
    
    @classmethod
    def get_vps(cls, nlpdoc, mytoken, ignore_tags=[]):
        '''
        Modified from code found at:
        https://github.com/explosion/spaCy/discussions/4338#discussioncomment-185327
        This version of the function is not as greedy in compiling a VP as the
        original code; the original version treats a VP as VP -> (NP | PP | VP),
        but for the purposes of feedback analysis a VP should be as small as possible.

        @param nlpdoc: spaCy Document.
        @param mytoken: Regex for specifying a token to parse out. Leave blank to return all possible verb phrases.
        @param ignore_tags: POS tags to not include in the verb phrases.
        @return: Verb phrases as a list.
        '''

        mylist = []
        i = 0

        patt = re.compile(mytoken)

        while i < len(nlpdoc):
            token = nlpdoc[i]
            word = token.lemma_.lower()
            ignore_verb = False

            if token.pos_ in ['VERB', 'AUX'] and \
            word not in string.punctuation:
                # get children on verb/aux/adp
                nodechild = token.children
                getchild1 = []
                getchild2 = []

                # iterate over the children
                for child in nodechild:
                    getchild1.append(child)
                    # get children of children
                    listchild = list(child.children)

                    for grandchild in listchild:
                        getchild2.append(grandchild)

                # check if Spacy is a children or a children of a children
                test1 = [patt.search(tok.lemma_) for tok in getchild1]
                test2 = [patt.search(tok.lemma_) for tok in getchild2]

                # if YES, then parse the VP
                if any(test1) or any(test2):
                    fulltok = [token]
                    myiter = token
                    # the VP can actually start a bit before the VERB, so we look for the leftmost AUX/VERBS
                    candidates = [lefty for lefty in token.lefts]
                    candidates = [lefty for lefty in candidates if lefty.pos_ in ['AUX', 'VERB', 'PART']]

                    # if we find one, then we start concatenating the tokens from there
                    if candidates and candidates[0].dep_ != 'ccomp':
                        fulltok = [candidates[0]]
                        myiter = candidates[0]

                    try:
                        # do not start creating a verb phrase if it's just an ADJ
                        if len(fulltok) == 1 and \
                        fulltok[0].pos_ in ('ADJ'):
                            ignore_verb = True

                        else:
                            while myiter.nbor().pos_ in ['VERB', 'PART', 'ADV', 'AUX', 'ADJ', 'PART']:
                                fulltok.append(myiter.nbor())
                                myiter = myiter.nbor()
                                i += 1

                                if len(ignore_tags) and \
                                myiter.nbor().tag_ in ignore_tags:
                                    ignore_verb = True

                    except IndexError:
                        pass

                    if not ignore_verb:
                        mylist.append(fulltok)

            i += 1

        return mylist
    
    @classmethod
    def extract_wh(cls, text):
        '''
        Extracts question words from given text.

        @param text: Text string.
        @return: Found text string, otherwise an empty string.
        '''

        m = re.search(r'\b(' + '|'.join(cls.wh_regex) + r')\b', text, re.I)

        if m is not None:
            grp = m.group(0).lower()

            if grp in PhraseExtractor.wh_map: grp = PhraseExtractor.wh_map[grp]

            return grp
        
        return ''