import spacy
import classes
import torch


def check_sentences(spacy_doc, dimi, context_window=2):
    possible_moralization = []

    sentences = list(spacy_doc.sents)
    for i, sentence in enumerate(sentences):
        dimi_words = []
        for token in sentence:
            if token.lemma_ in dimi:
                dimi_words.append({
                    'lemma': token.lemma_,
                    'text': token.text
                })
        if len(dimi_words) > 0:
            precontext = [
                token.text
                for token in sentences[-context_window:i]
            ]
            postcontext = [
                token.text
                for token in sentences[i+1:][:context_window]
            ]
            metadata = classes.MetaData('', '', '', '', '')
            poss_moral = classes.PossibleMoralizationDimi(
                precontext_len=context_window,
                postcontext_len=context_window,
                metadata=metadata,
                dimi_words=dimi_words
            )

            poss_moral.focus_sentence = sentence.text
            poss_moral.precontext = precontext
            poss_moral.postcontext = postcontext

            possible_moralization.append(poss_moral)

    return possible_moralization


def scan_text_dimi(
    text, dimi,
    model='de_core_news_lg',
    stop_at_paragraph=False,
    context_window=2
):

    possible_moralization = []

    if stop_at_paragraph:
        paragraphs = text.split('\n')
        for paragraph in paragraphs:
            if paragraph == '':
                break
            instances = scan_text_dimi(
                paragraph, model, dimi
            )
            possible_moralization.extend(instances)

    else:
        nlp = spacy.load(model)
        doc = nlp(text)

        possible_moralization = check_sentences(
            doc, dimi, context_window
        )

    return possible_moralization


def bert_classification(
    possible_moralization, tokenizer, classifier, device,
    **kwargs
):

    if isinstance(possible_moralization, classes.PossibleMoralization):
        possible_moralization_str = possible_moralization.full_text
    else:
        possible_moralization_str = possible_moralization

    tokens = tokenizer.encode(possible_moralization_str, kwargs)
    attention_mask = [int(token > 0) for token in tokens]

    tokens = torch.tensor(tokens).unsqueeze(0) # Batch size 1
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)

    # Make the prediction
    with torch.no_grad():
        outputs = classifier(tokens, attention_mask=attention_mask)
        logits = outputs.logits

    return logits


def scan_text_dimi_modelled(
    text, dimi, classifier, tokenizer, device,
    **kwargs
):

    possible_moralizations = scan_text_dimi(
        text, dimi, kwargs
    )

    moralizations_modelled = []

    for poss_moral in possible_moralizations:
        logits = bert_classification(
            poss_moral, tokenizer, classifier, kwargs
        )

        modelled_moral = classes.PossibleMoralizationModelled(
            poss_moral, logits
        )

        moralizations_modelled.append(modelled_moral)

    return moralizations_modelled
