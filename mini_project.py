from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
model.fit(Xtr, ytr)

print("Test accuracy:", model.score(Xte, yte))
