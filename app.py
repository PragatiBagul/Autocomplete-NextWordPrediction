import streamlit as st

from NextWordPredictor import NextWordPredictor
nextWordPredictor = NextWordPredictor()
nextWordPredictor.load(path = './data')
TOP_K = 5
st.set_page_config(page_title="Autocomplete Engine", layout="centered")
st.title("⌨️ Autocomplete + Next Word Prediction")

sentence = st.text_input(
    "Type here",
    placeholder="when ar"
)

# -------------------------------
# Dispatcher
# -------------------------------
def get_suggestions(sentence):
    if not sentence.strip():
        return []

    ends_with_space = sentence.endswith(" ")

    tokens = sentence.strip().split()

    # CASE 1: Next-word prediction
    if ends_with_space:
        context = tokens
        prefix = ""
        mode = "next_word"

    # CASE 2: Word completion
    else:
        context = tokens[:-1]
        prefix = tokens[-1]
        mode = "completion"

    return mode, context, prefix


# -------------------------------
# Prediction
# -------------------------------
def predict(mode, context, prefix, top_k=5):

    suggestions = []

    # Word completion
    if mode == "completion":
        st.write(f"=============================Completing {prefix}")
        answer = nextWordPredictor.predict(context, prefix)
        print(f"Word Completion : {answer}")
        suggestions.extend(answer)
        # candidates = trie.starts_with(prefix)
        #
        # for w in candidates:
        #     score = lm.probability(context, w)
        #     suggestions.append((w, score))

    # Next-word prediction
    elif mode == "next_word":
        st.write(f"=============================Predicting {context}")
        # candidates = lm.get_next_word_candidates(context)
        #
        # for w in candidates:
        #     score = lm.probability(context, w)
        #     suggestions.append((w, score))
        answer = nextWordPredictor.predict_next(context)
        print(f"Next Word Prediction : {answer}")
        suggestions.extend(answer)

    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions[:top_k]


# -------------------------------
# UI Rendering
# -------------------------------
if sentence:
    ends_with_space = sentence.endswith(" ")
    mode = ""
    if ends_with_space:
        mode = "next_word"
    else:
        mode = "completion"
    mode, context, prefix = get_suggestions(sentence)
    results = predict(mode, context, prefix, TOP_K)
    print(f"Results : {results}")
    st.subheader(
        "Word Completion" if mode == "completion" else "Next Word Prediction"
    )

    for i, (word, score) in enumerate(results, 1):
        st.write(f"**{i}. {word}**  (`{score:.4f}`)")