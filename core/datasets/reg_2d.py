import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer

def load_openml(name=None,data_id=None,version="active",target=None,as_frame=True):
    if name is None and data_id is None:
        raise ValueError("Specify name or data_id.")

    ds=fetch_openml(
        name=name,
        data_id=data_id,
        version=version,
        as_frame=as_frame,
        parser="auto"
    )

    X=ds.data.copy()
    y=ds.target.copy() if target is None else X.pop(target)

    X=X.replace(r"^\s*$",np.nan,regex=True)

    num=[]
    cat=[]
    feature_types={}
    unique_counts={}

    for c in X.columns:
        if is_numeric_dtype(X[c]):
            num.append(c)
            feature_types[c]="numeric"
        else:
            cat.append(c)
            feature_types[c]="categorical"
        unique_counts[c]=int(X[c].nunique(dropna=True))

    unique_labels=np.unique(y).tolist()

    def make_preprocessor(scale_numeric=True,encode="onehot"):
        t=[]

        if num:
            steps=[("imputer",SimpleImputer(strategy="median"))]
            if scale_numeric:
                steps.append(("scaler",StandardScaler()))
            t.append(("num",Pipeline(steps),num))

        if cat:
            steps=[("imputer",SimpleImputer(strategy="most_frequent"))]

            if encode=="onehot":
                steps.append(("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False)))
            elif encode=="ordinal":
                steps.append(("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)))
            elif encode is not None:
                raise ValueError("encode must be 'onehot','ordinal', or None.")

            t.append(("cat",Pipeline(steps),cat))

        return ColumnTransformer(t,remainder="drop")

    details={
        "feature_types":feature_types,
        "unique_counts":unique_counts,
        "unique_labels":unique_labels
    }

    return X.to_numpy(dtype=object),np.asarray(y),make_preprocessor,details