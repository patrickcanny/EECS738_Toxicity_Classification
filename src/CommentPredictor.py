import tools

class CommentPredictor:

    def __init__(self, tools):
        self.tools = tools
        self.word_weight_dict = {}
        self.predictions = None

        self.initializeWeights()
        self.predictAll()

    def initializeWeights(self):
        words = self.getEveryWord()
        toxic_words = self.getToxicWords()
        non_toxic_words = self.getNonToxicWords()
        stop_words = self.tools.getStops()
        for word in words:
            weight = 1.0
            if word in non_toxic_words:
                weight *= 0.25
            if word in toxic_words:
                weight *= -0.75
            if word in stop_words:
                weight *= 0.0
            self.word_weight_dict[word] = weight

    def predictAll(self):
        scores = []
        comments = self.tools.getAllComments()
        predict = {}
        for comment in comments:
            scores.append(self.predictForComment(comment))
        predict = {'target':scores, 'comment_text':comments}
        self.predictions = pd.DataFrame(predict)

    def predictForComment(self, comment):
        score = 0.0
        for word in comment:
            score += self.word_weight_dict[word]
        return score

    def getEveryWord(self):
        comments = self.tools.getAllComments()
        return self.getAllWords(comments)

    def getToxicWords(self):
        comments = self.tools.getMyToxicComments()
        return self.getAllWords(comments)

    def getNonToxicWords(self):
        comments = self.tools.getMyNonToxicComments()
        return self.getAllWords(comments)

    def getAllWords(self, comments):
        s = set()
        for comment in comments:
            for word in comment.split():
                s.append(word)
        return s

