1. OCR pipeline for pdfs containing handwritten letters from the 1800s (among other things -- but those will be the hardest by far).
2. use LLMs to check through these, perhaps in a few ways.
    1. to handle difficult OCR tasks, do it with multiple OCR tools and then reconcile.
    2. in cases where pieces of the handwriting are missing, use AI vision models to guess what was there and/or LLMs to guess the missing words.
    3. have the AI do image recognition/description of the images on the page, and save that (perhaps as further metadata).
3. make this searchable, perhaps even with a LLM (maybe with RAG) for natural language interactions and semantic search.