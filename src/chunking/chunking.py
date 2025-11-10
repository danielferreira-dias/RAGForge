from sentence_transformers import SentenceTransformer

class BaseChunker:
    def __init__(self, model_name : SentenceTransformer ='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def cunk_document(self):
        pass

class ContextAwareChunking(BaseChunker):
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)

class LateChunking(BaseChunker):
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
