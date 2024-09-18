from collections import defaultdict
from collections import namedtuple
from typing import List
from typing import Optional

import lifelines
import matplotlib.pyplot as plt
import numpy
import pandas
import tqdm
import seaborn as sns
from sklearn.preprocessing import StandardScaler


LassoCOXTABLResult = namedtuple(
    'LassoCOXTABLResult',
    [
        'CoxPHFitters',
        'l1_to_names_to_coefs',
        'data_cols',
        'artificial_cols',
    ]
)


FeatureSelectionResult = namedtuple(
    'FeatureSelectionResult',
    [
        'thresholds',
        'fdps',
        'threshold',
        'selected_features',
    ]
)


def lasso_coxtabl_on_df(
    df,
    event_col: str = 'death',
    duration_col: str = 'days_to_death',
    data_cols: Optional[list] = None,
    l1s_to_check: Optional[list] = None,
    l2: float = 0.1,
    iters: int = 100,
    bootstrap_sample: float = 0.5,
    random_seed: int = 42,
    standardize: bool = True,
    permuted_prefix: str = 'ART',
    cox_kwargs: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
):
    if data_cols is None:
        data_cols = [
            col for col in df.columns if col not in [event_col, duration_col]
        ]
    if l1s_to_check is None:
        l1s_to_check = numpy.logspace(-1, -1.5, 10)
    if cox_kwargs is None:
        cox_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    coxes = []
    l1_to_names_to_coefs = defaultdict(lambda: defaultdict(list))
    current_random_seed = random_seed

    for _ in tqdm.tqdm(range(iters)):
        failed = True
        while failed:
            try:
                artificial_features = get_permuted_df(
                    df=df[data_cols],
                    permuted_prefix=permuted_prefix,
                    random_seed=current_random_seed,
                )
                current_random_seed += 1
                artificial_features_names = list(artificial_features.columns)
                cox_cols = [event_col, duration_col] + data_cols + artificial_features_names
                cox_features_cols = data_cols + artificial_features_names
                full_coxtabl_df = pandas.concat([df, artificial_features], axis=1)
                iter_df = full_coxtabl_df.sample(frac=bootstrap_sample, random_state=current_random_seed)
                current_random_seed += 1

                for l1 in l1s_to_check:
                    if standardize:
                        iter_df = iter_df.copy()
                        scaler = StandardScaler()
                        iter_df[cox_features_cols] = scaler.fit_transform(iter_df[cox_features_cols].values)
                    penalizer, l1_ratio = pen_l1_r_from_l1_l2(l1=l1, l2=l2)
                    current_cphf = lifelines.CoxPHFitter(
                        penalizer=penalizer,
                        l1_ratio=l1_ratio,
                        **cox_kwargs)
                    current_cphf.fit(
                        df=iter_df[cox_cols],
                        duration_col=duration_col,
                        event_col=event_col,
                        **fit_kwargs,
                    )
                    summary = current_cphf.summary['coef']
                    for row in summary.index:
                        l1_to_names_to_coefs[l1][row].append(summary[row])
                failed = False
            except lifelines.exceptions.ConvergenceError:
                pass
            except lifelines.exceptions.ConvergenceWarning:
                pass
            except scipy.linalg.LinAlgWarning:
                pass


    result = LassoCOXTABLResult(
        CoxPHFitters=coxes,
        l1_to_names_to_coefs=l1_to_names_to_coefs,
        data_cols=data_cols,
        artificial_cols=artificial_features_names,
    )
    return result


def compute_acceptance_per_l1_df(
    result: LassoCOXTABLResult,
    threshold: float = 0.01,
):
    l1s = []
    l1s_dict = defaultdict(list)
    for l1, names_to_coefs in result.l1_to_names_to_coefs.items():
        l1s.append(l1)
        for feature in names_to_coefs:
            l1s_dict[feature].append((numpy.abs(names_to_coefs[feature]) > threshold).mean())
    acceptance_df = pandas.DataFrame(l1s_dict)
    acceptance_df.index = l1s
    acceptance_df.index.name = 'l1'
    acceptance_df.columns.name = 'Features'
    return acceptance_df


def plot_acceptance_matrix(acceptance_matrix, permuted_prefix: str = 'ART'):
    feature_relevance = (acceptance_matrix.T * (acceptance_matrix.shape[0] - numpy.arange(acceptance_matrix.shape[0]))).sum(
        axis=1
    ).values.argsort()
    all_features = list(acceptance_matrix.columns)
    features_sorted = [all_features[i] for i in feature_relevance]
    feature_colors = ['r' if col.startswith(permuted_prefix) else 'g' for col in features_sorted]
    fig = sns.clustermap(
        acceptance_matrix[features_sorted],
        col_colors=feature_colors,
        row_cluster=False,
        col_cluster=False,
        xticklabels=features_sorted,
        cbar_pos=(1., 0.3, 0.01, 0.3),
        cbar_kws={'label': 'Acceptance ratio'},
        cmap='Greens',
        figsize=(0.25 * acceptance_matrix.shape[1], 0.75 * acceptance_matrix.shape[0]),
    )
    fig.ax_col_dendrogram.set_visible(False)
    fig.ax_row_dendrogram.set_visible(False)


def compute_fdp_for_threshold(
    max_acceptance_df, 
    threshold,
    data_cols: List[str],
    artificial_cols: List[str],
):
    nom = (max_acceptance_df.loc[data_cols] > threshold).values.sum() + 1
    denom = max((max_acceptance_df.loc[artificial_cols] > threshold).values.sum(), 1)
    return nom / denom


def feature_selection(
    acceptance_per_l1_df, 
    data_cols: List[str],
    artificial_cols: List[str],
    steps: int = 100,
):
    max_acceptance_df = acceptance_per_l1_df.max(axis=0).to_frame()
    thresholds = numpy.linspace(0, 1, steps)
    fdps = [compute_fdp_for_threshold(
        max_acceptance_df, 
        t,
        data_cols,
        artificial_cols,
    ) for t in thresholds]
    threshold = thresholds[numpy.array(fdps).argmin()]
    norm_selector = (max_acceptance_df > threshold).loc[data_cols]
    selected_features = max_acceptance_df.loc[data_cols][norm_selector.values].index
    return FeatureSelectionResult(
            thresholds,
            fdps,
            threshold,
            selected_features,
        )


def plot_feature_selection_result(feature_selection_result):
    plt.plot(feature_selection_result.thresholds, feature_selection_result.fdps)
    plt.vlines(feature_selection_result.threshold, ymin=0, ymax=max(feature_selection_result.fdps), color='r', linestyles='dashed')
    plt.title(f'FDP vs Thresholds for feature selection\nSelected threshold (in red): {round(feature_selection_result.threshold, 4)}')
    plt.xlabel('FDP')
    plt.ylabel('Thresholds')


def get_permuted_df(
    df, 
    permuted_prefix: str = 'ART',
    random_seed: int = 42,
):
    new_df = df.copy()
    numpy.random.seed(random_seed)
    for col in new_df.columns:
        new_df[col] = numpy.random.permutation(df[col].values)
    new_df.columns = [f'{permuted_prefix}:{col}' for col in new_df.columns]
    return new_df


def pen_l1_r_from_l1_l2(l1: float, l2: float):
    penalizer = l1 + l2
    l1_ratio = l1 / (l1 + l2)
    return penalizer, l1_ratio
