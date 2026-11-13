from StanceDetectorModel import StanceDetector, Tfidf_Nn, tfidf
from StanceDetectorModel import x_train, y_train, x_dev, y_dev, x_test, y_test
from sklearn.metrics import classification_report

model = Tfidf_Nn()
stance_detector = StanceDetector(model, tfidf)
history = stance_detector.fit(x_train, y_train, x_dev, y_dev, epochs=100, verbose=1)

predictions, probabilities, real = stance_detector.get_predictions(x_test, y_test)
print(classification_report(real, predictions, digits=4,
                            target_names=['support', 'deny', 'query', 'comment']))

print(stance_detector.predict_new(model, 'example.txt'))
