from collections import defaultdict
from collections import namedtuple
from typing import List
from typing import Optional
from typing import Union 
import warnings

import lifelines
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy
import seaborn as sns
import tqdm
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
    lambdas: Optional[Union[list, int]] = 10,
    alpha: float = 0.95,
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
    
    if standardize:
        df = df.copy()
        scaler = StandardScaler()
        df[data_cols] = scaler.fit_transform(df[data_cols].values)

    if isinstance(lambdas, int):
        lambdas = get_optimal_lambdas(
            times=df[duration_col].values,
            events=df[event_col].values,
            x=df[data_cols].values,
            alpha=alpha,
            n_lambda=lambdas,
        )

    if cox_kwargs is None:
        cox_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    coxes = []
    l1_to_names_to_coefs = defaultdict(lambda: defaultdict(list))
    current_random_seed = random_seed
    
    warnings.filterwarnings('ignore')
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

                for lambda_ in lambdas:
                    print(lambda_)
                    current_cphf = lifelines.CoxPHFitter(
                        penalizer=lambda_,
                        l1_ratio=alpha,
                        **cox_kwargs)
                    current_cphf.fit(
                        df=iter_df[cox_cols],
                        duration_col=duration_col,
                        event_col=event_col,
                        **fit_kwargs,
                    )
                    summary = current_cphf.summary['coef']
                    for row in summary.index:
                        l1_to_names_to_coefs[lambda_][row].append(summary[row])
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
    feature_relevance = acceptance_matrix.max(axis=0).values.argsort()
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
    nom = (max_acceptance_df.loc[artificial_cols] > threshold).values.sum() + 1
    denom = max((max_acceptance_df.loc[data_cols] > threshold).values.sum(), 1)
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
    selected_features = numpy.array(max_acceptance_df.loc[data_cols][norm_selector.values].index)
    selected_features = list(selected_features[max_acceptance_df.loc[data_cols][norm_selector.values].values[:, 0].argsort()])
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


def get_optimal_lambdas(
        times,
        events,
        x,
        alpha,
        n_lambda,
):
    aux_cox_vector = compute_cox_aux_vector(times, events)
    lambda_vectors = numpy.matmul(x.T, aux_cox_vector) / (x.shape[0] * alpha)
    lambda_max = numpy.abs(lambda_vectors).max()
    return numpy.geomspace(lambda_max / 20, lambda_max, n_lambda)


def compute_cox_aux_vector(
        times, 
        events,
        random_seed=42, 
        survival_time_unit=1,
):
    numpy.random.seed(random_seed)
    times_perturbed = times + numpy.random.random((len(times),)) / (2 * survival_time_unit)
    time_sorting = times_perturbed.argsort()

    times_sorted = times[time_sorting]
    events_sorted = events[time_sorting]

    n = len(events_sorted)
    r_powers_for_events = numpy.maximum(events_sorted * (n - numpy.arange(0, n)), 1) * events_sorted
    inv_r_powers_for_events = 1 / r_powers_for_events
    inv_r_powers_for_events[numpy.logical_not(events_sorted)] = 0

    result = events_sorted.copy().astype('float')
    result[1:] -= inv_r_powers_for_events.cumsum()[:-1]

    return result[time_sorting.argsort()]

