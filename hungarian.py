"""
Hungarian Assignment Algorithm for RF-Project Matching
A clean implementation that uses only RF preferences to find optimal assignments.
Now supports reading preferences from JSON files.
"""

import json
from typing import Dict, List, Set, Tuple


class HungarianAssignment:
    """
    Implementation of the Hungarian algorithm for optimal assignment based on RF preferences only.
    """

    def __init__(self, rfs: List[str], projects: List[str]):
        """
        Initialize the Hungarian assignment algorithm.

        Args:
            rfs: List of RF names
            projects: List of project names
        """
        self.rfs = rfs
        self.projects = projects
        self.n = len(rfs)

        if len(projects) != self.n:
            raise ValueError("Number of RFs and projects must be equal")

    def solve(self, rf_preferences: Dict[str, List[List[str]]]) -> List[Tuple[str, str]]:
        """
        Find optimal assignment using the Hungarian algorithm based on RF preferences.

        Algorithm:
        1. Create a cost matrix from RF preferences.
        2. Apply the Hungarian algorithm to find the optimal assignment.

        Args:
            rf_preferences: Dictionary mapping RF names to preference lists (with ties)

        Returns:
            List of tuples representing optimal assignments (rf, project)
        """
        # Create cost matrix from RF preferences
        cost_matrix = self._create_cost_matrix(rf_preferences)

        # Run Hungarian algorithm
        assignment = self._hungarian_algorithm(cost_matrix)

        # Convert back to string names
        return [(self.rfs[rf_idx], self.projects[proj_idx]) for rf_idx, proj_idx in assignment]

    def _create_cost_matrix(self, rf_preferences: Dict[str, List[List[str]]]) -> List[List[int]]:
        """
        Create cost matrix from RF preferences.
        Lower costs represent higher preferences.

        Algorithm:
        1. Initialize a cost matrix with high default costs.
        2. For each RF, set costs based on their preferences.
        3. If a project is unranked, assign a very high cost.
        4. If a project is ranked, assign incremental costs based on preference tiers.
        5. Return the cost matrix.

        Args:
            rf_preferences: Dictionary mapping RF names to preference lists (with ties)

        Returns:
            Cost matrix where cost[i][j] is the cost of assigning RF i to project j
        """
        cost_matrix: List[List[int]] = []

        for rf in self.rfs:
            rf_costs: List[int] = [self.n + 1] * self.n  # Default high cost for unranked projects

            if rf in rf_preferences:
                current_cost = 1
                for tier in rf_preferences[rf]:
                    for project in tier:
                        if project in self.projects:
                            proj_idx = self.projects.index(project)
                            rf_costs[proj_idx] = current_cost
                    current_cost += len(tier)

            cost_matrix.append(rf_costs)

        return cost_matrix

    def _hungarian_algorithm(self, cost_matrix: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Core Hungarian algorithm implementation.

        Algorithm:
        1. Subtract row minima from each row.
        2. Subtract column minima from each column.
        3. Find minimum number of lines to cover all zeros.
        4. If the number of lines equals n, return the assignment.
        5. If not, improve the matrix by finding minimum vertex cover and adjusting values.
        6. Repeat until an assignment is found.

        Args:
            cost_matrix: Square matrix of assignment costs

        Returns:
            List of tuples representing optimal assignments (rf_idx, project_idx)
        """
        # Work with a copy to avoid modifying original
        matrix = [row[:] for row in cost_matrix]

        # Step 1: Subtract row minima
        for i in range(self.n):
            row_min = min(matrix[i])
            for j in range(self.n):
                matrix[i][j] -= row_min

        # Step 2: Subtract column minima
        for j in range(self.n):
            col_min = min(matrix[i][j] for i in range(self.n))
            for i in range(self.n):
                matrix[i][j] -= col_min

        # Step 3: Find minimum number of lines to cover all zeros
        while True:
            # Try to find maximum matching
            assignment = self._find_maximum_matching(matrix)

            if len(assignment) == self.n:
                return assignment

            # Step 4: Improve the matrix
            matrix = self._improve_matrix(matrix, assignment)

    def _find_maximum_matching(self, matrix: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Find maximum matching using zeros in the matrix.

        Algorithm:
        1. Find all positions with zero cost.
        2. Use a greedy approach to find maximum matching.
        3. Return the list of assignments.
        4. If no assignment is possible, return an empty list.

        Args:
            matrix: Cost matrix with zeros representing possible assignments

        Returns:
            List of assignments (rf_idx, project_idx)
        """
        # Find all zero positions
        zeros = [(i, j) for i in range(self.n) for j in range(self.n) if matrix[i][j] == 0]

        # Try to find maximum matching using greedy approach with backtracking
        return self._find_matching_greedy(zeros)

    def _find_matching_greedy(self, zeros: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Find maximum matching using greedy approach.

        Algorithm:
        1. Sort zeros by number of zeros in their row + column (prefer constrained positions).
        2. Iterate through sorted zeros and build assignment.
        3. Use sets to track used rows and columns to avoid conflicts.
        4. Return the list of assignments.
        5. If no assignment is possible, return an empty list.

        Args:
            zeros: List of (row, col) positions with zero cost

        Returns:
            List of assignments
        """

        # Sort zeros by number of zeros in their row + column (prefer constrained positions)
        def count_zeros_in_row_col(pos: Tuple[int, int]) -> int:
            row, col = pos
            row_zeros = sum(1 for r, _ in zeros if r == row)
            col_zeros = sum(1 for _, c in zeros if c == col)
            return row_zeros + col_zeros

        zeros.sort(key=count_zeros_in_row_col)

        assignment: List[Tuple[int, int]] = []
        used_rows: Set[int] = set()
        used_cols: Set[int] = set()

        for row, col in zeros:
            if row not in used_rows and col not in used_cols:
                assignment.append((row, col))
                used_rows.add(row)
                used_cols.add(col)

        return assignment

    def _improve_matrix(
        self, matrix: List[List[int]], current_assignment: List[Tuple[int, int]]
    ) -> List[List[int]]:
        """
        Improve the matrix by finding minimum vertex cover and adjusting values.

        Algorithm:
        1. Find minimum vertex cover using König's theorem.
        2. Mark rows and columns based on current assignment.
        3. Find minimum uncovered element.
        4. Adjust matrix:
              - Subtract from uncovered elements.
              - Add to elements covered twice.
        5. Return the improved matrix.

        Args:
            matrix: Current cost matrix
            current_assignment: Current partial assignment

        Returns:
            Improved matrix
        """
        # Find minimum vertex cover using König's theorem
        assigned_rows = {row for row, _ in current_assignment}

        # Mark rows and columns
        marked_rows: Set[int] = set()
        marked_cols: Set[int] = set()

        # Mark unassigned rows
        for i in range(self.n):
            if i not in assigned_rows:
                marked_rows.add(i)

        # Iteratively mark columns and rows
        changed = True
        while changed:
            changed = False

            # Mark columns that have zeros in marked rows
            for row in marked_rows:
                for col in range(self.n):
                    if matrix[row][col] == 0 and col not in marked_cols:
                        marked_cols.add(col)
                        changed = True

            # Mark rows that are assigned to marked columns
            for row, col in current_assignment:
                if col in marked_cols and row not in marked_rows:
                    marked_rows.add(row)
                    changed = True

        # Minimum vertex cover consists of:
        # - unmarked rows
        # - marked columns
        cover_rows = {i for i in range(self.n) if i not in marked_rows}
        cover_cols = marked_cols

        # Find minimum uncovered element
        min_uncovered = float("inf")
        for i in range(self.n):
            for j in range(self.n):
                if i not in cover_rows and j not in cover_cols:
                    min_uncovered = min(min_uncovered, matrix[i][j])

        # Adjust matrix
        new_matrix = [row[:] for row in matrix]
        for i in range(self.n):
            for j in range(self.n):
                if i not in cover_rows and j not in cover_cols:
                    # Subtract from uncovered elements
                    new_matrix[i][j] -= int(min_uncovered)
                elif i in cover_rows and j in cover_cols:
                    # Add to elements covered twice
                    new_matrix[i][j] += int(min_uncovered)

        return new_matrix

    def calculate_total_cost(
        self, assignment: List[Tuple[str, str]], rf_preferences: Dict[str, List[List[str]]]
    ) -> int:
        """
        Calculate total cost of an assignment.

        Algorithm:
        1. Create cost matrix from RF preferences.
        2. Sum costs for each (rf, project) assignment based on the cost matrix.
        3. Return the total cost.

        Args:
            assignment: List of (rf, project) assignments
            rf_preferences: RF preferences used to calculate costs

        Returns:
            Total cost of the assignment
        """
        total_cost = 0
        cost_matrix = self._create_cost_matrix(rf_preferences)

        for rf, project in assignment:
            rf_idx = self.rfs.index(rf)
            proj_idx = self.projects.index(project)
            total_cost += cost_matrix[rf_idx][proj_idx]

        return total_cost


def load_preferences_from_file(
    filename: str,
) -> Tuple[List[str], List[str], Dict[str, List[List[str]]]]:
    """
    Load preferences from a JSON file for Hungarian Assignment algorithm.

    Expected JSON format:
    {
        "proposers": ["RF1", "RF2", ...],
        "receivers": ["Project1", "Project2", ...],
        "proposer_preferences": {
            "RF1": [["Project1"], ["Project2", "Project3"]],
            ...
        },
        "receiver_preferences": {
            "Project1": [["RF1", "RF2"], ["RF3"]],
            ...
        }
    }

    Note: This algorithm only uses proposer_preferences (RF preferences),
    but loads from the same file format as the Gale-Shapley algorithm.

    Args:
        filename: Path to the JSON file containing preferences

    Returns:
        Tuple of (rfs, projects, rf_preferences)

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        KeyError: If required keys are missing from the JSON
        ValueError: If the data format is invalid
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Preferences file '{filename}' not found") from exc
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file '{filename}': {e.msg}", e.doc, e.pos)

    # Validate required keys (only need proposers and their preferences for Hungarian algorithm)
    required_keys = ["proposers", "receivers", "proposer_preferences"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Missing required keys in preferences file: {missing_keys}")

    rfs: list[str] = data["proposers"]  # RFs are the proposers
    projects: list[str] = data["receivers"]  # Projects are the receivers
    rf_preferences: Dict[str, List[List[str]]] = data[
        "proposer_preferences"
    ]  # Only need RF preferences

    # Validate preference structure
    for rf, prefs in rf_preferences.items():
        for _, tier in enumerate(prefs):
            for choice in tier:
                if choice not in projects:
                    raise ValueError(f"Invalid project '{choice}' in preferences for '{rf}'")

    return rfs, projects, rf_preferences


def display_results(
    result: List[Tuple[str, str]],
    rfs: List[str],
    projects: List[str],
    rf_preferences: Dict[str, List[List[str]]],
    total_cost: int,
) -> None:
    """
    Display the assignment results and unmatched participants.

    Args:
        result: List of (RF, Project) tuples
        rfs: List of all RFs
        projects: List of all projects
        rf_preferences: RF preferences for calculating satisfaction
        total_cost: Total cost of the assignment

    Returns:
        None
    """
    print("Optimal assignment (RF -> Project):")
    for rf, project in sorted(result):
        # Find preference rank for this assignment
        rank = "Unranked"
        if rf in rf_preferences:
            current_rank = 1
            for tier in rf_preferences[rf]:
                if project in tier:
                    rank = f"Rank {current_rank}"
                    break
                current_rank += len(tier)

        print(f"  {rf} -> {project} ({rank})")

    # Show unmatched participants
    matched_rfs = {rf for rf, _ in result}
    matched_projects = {project for _, project in result}

    unmatched_rfs = [rf for rf in rfs if rf not in matched_rfs]
    unmatched_projects = [project for project in projects if project not in matched_projects]

    if unmatched_rfs:
        print(f"\nUnmatched RFs: {', '.join(unmatched_rfs)}")
    if unmatched_projects:
        print(f"Unmatched Projects: {', '.join(unmatched_projects)}")

    print(f"\nAssignment Summary: {len(result)} assignments found")
    print(f"Total Cost: {total_cost} (lower is better)")


def main():
    """
    RF-Project optimal assignment using Hungarian algorithm with file-based preferences.
    """
    preferences_file = "preferences.json"
    rfs = None
    projects = None
    rf_preferences = None

    # Create sample file if it doesn't exist
    try:
        rfs, projects, rf_preferences = load_preferences_from_file(preferences_file)
        print(f"Loaded preferences from {preferences_file}")
    except FileNotFoundError:
        print(f"Preferences file '{preferences_file}' not found.")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error loading preferences: {e}")
        return

    # Create matcher and solve
    try:
        if rfs is None or projects is None or rf_preferences is None:
            raise ValueError("Preferences data is incomplete or missing")
        matcher = HungarianAssignment(rfs, projects)
        result = matcher.solve(rf_preferences)
        total_cost = matcher.calculate_total_cost(result, rf_preferences)
        display_results(result, rfs, projects, rf_preferences, total_cost)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"Error during assignment: {e}")


if __name__ == "__main__":
    main()
