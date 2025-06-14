import polars as pl
from polars._typing import EngineType
from typing import List, Any
from itertools import combinations as itertools_combinations
import time


def generate_combinations(
    combination_size: int, num_of_combinations: int, items: List[str]
) -> List[List[str]]:
    """
    Generates combinations of a specific size from a list and returns a limited number of them.

    Args:
        combination_size: The size of each combination (number of elements in each combination).
        num_of_combinations: The maximum number of combinations to return.
        items: The list of items to choose combinations from.

    Returns:
        A list containing up to 'c' combinations. Each combination is a list of 'e' elements.
        Returns an empty list if 'c' is non-positive, 'e' is negative,
        or if no combinations of the given size can be formed.
    """
    if (
        num_of_combinations <= 0
    ):  # If no combinations are requested, or a non-positive number.
        return []
    if combination_size < 0:  # Combination size cannot be negative.
        return []
    # If e is 0, the only combination is an empty list.
    # itertools.combinations handles e > len(a) by returning an empty iterator.

    # itertools.combinations returns an iterator of tuples.
    comb_iter = itertools_combinations(items, combination_size)

    result_combinations: List[List[Any]] = []
    count = 0

    for combo_tuple in comb_iter:
        if count >= num_of_combinations:
            break
        result_combinations.append(list(combo_tuple))  # Convert tuple to list
        count += 1

    return result_combinations


def extract_column_combinations(
    n_combinations: int, column_list: List[str]
) -> List[List[str]]:
    number_of_elements = 2  # start with 2 elements
    total_collected_combinations = 0
    combined_list = []
    while total_collected_combinations < n_combinations:
        new_combs = generate_combinations(
            number_of_elements,
            n_combinations - total_collected_combinations,
            column_list,
        )
        combined_list.extend(new_combs)
        total_collected_combinations += len(new_combs)
        number_of_elements += 1
    return combined_list


def multi_entropy(ldf: pl.LazyFrame, n: int, engine: EngineType) -> None:
    """
    Calculates entropy for n combinations of columns.

    :param ldf: dataframe to use
    :param n: number of other column combination
    :param engine: polars engine type to use
    """
    cols = ldf.collect_schema().names()
    columns_combinations = extract_column_combinations(n, cols)

    entropy_exprs = [
        pl.concat_str(columns, separator="_")
        .unique_counts()
        .entropy()
        .alias(f"{'_'.join(columns)}")
        for columns in columns_combinations
    ]
    with_entropy_calculations = ldf.select(entropy_exprs)
    print(with_entropy_calculations.collect(engine=engine))


if __name__ == "__main__":
    path = "example-data/stackoverflow_full.csv"
    df = pl.scan_csv(path).drop("id")

    start_time = time.time()
    print("running with in-memory engine...")
    multi_entropy(df, 1000, "in-memory")
    print(time.time() - start_time)

    print("running with streaming engine...")
    start_time = time.time()
    multi_entropy(df, 1000, "streaming")
    print(time.time() - start_time)
