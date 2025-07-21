"""
Gale-Shapley Algorithm with Ties and Partial Preferences
A clean implementation for stable matching problems with support for ties and partial preferences.
Now supports reading preferences from JSON files.
"""

import json
from typing import Dict, List, Optional, Tuple


class StableMatchingWithTies:
    """
    Implementation of the Gale-Shapley algorithm with support for ties and partial preferences.
    """

    def __init__(self, proposers: List[str], receivers: List[str]):
        """
        Initialize the stable matching algorithm.

        Args:
            proposers: List of proposer names (e.g., men, RFs)
            receivers: List of receiver names (e.g., women, projects)
        """
        self.proposers = proposers
        self.receivers = receivers
        self.n = len(proposers)

        if len(receivers) != self.n:
            raise ValueError("Number of proposers and receivers must be equal")

    def solve(
        self, proposer_prefs: Dict[str, List[List[str]]], receiver_prefs: Dict[str, List[List[str]]]
    ) -> List[Tuple[str, str]]:
        """
        Find stable matching using the Gale-Shapley algorithm.

        Algorithm:
        1. Convert string preferences to indexed preferences for efficient comparison.
        2. Create ranking matrices for proposers and receivers.
        3. Run the Gale-Shapley algorithm to find stable matchings.

        Args:
            proposer_prefs: Dictionary mapping proposer names to preference lists (with ties)
            receiver_prefs: Dictionary mapping receiver names to preference lists (with ties)

        Returns:
            List of tuples representing stable matchings (proposer, receiver)
        """
        # Convert string preferences to indexed preferences
        proposer_indexed_prefs = self._convert_to_indexed_prefs(proposer_prefs, self.receivers)
        receiver_indexed_prefs = self._convert_to_indexed_prefs(receiver_prefs, self.proposers)

        # Create ranking matrices for efficient comparison
        proposer_rankings = self._create_rankings(proposer_indexed_prefs)
        receiver_rankings = self._create_rankings(receiver_indexed_prefs)

        # Run the algorithm
        matches = self._gale_shapley(proposer_indexed_prefs, proposer_rankings, receiver_rankings)

        # Convert back to string names
        return [(self.proposers[p], self.receivers[r]) for p, r in matches]

    def _convert_to_indexed_prefs(
        self, prefs: Dict[str, List[List[str]]], target_list: List[str]
    ) -> List[List[List[int]]]:
        """
        Convert string-based preferences to index-based preferences.

        Algorithm:
        1. For each person, convert their preference tiers to indices based on the target list.
        2. Each tier is represented as a list of indices corresponding to the target list.

        Args:
            prefs: Dictionary mapping person names to preference lists (with ties)

        Returns:
            Indexed preference lists where each tier is a list of indices
        """
        indexed_prefs: List[List[List[int]]] = []
        if target_list == self.receivers:
            persons = self.proposers
        else:
            persons = self.receivers
        for person in persons:
            person_prefs: List[List[int]] = []
            for tier in prefs.get(person, []):
                tier_indices = [target_list.index(target) for target in tier]
                person_prefs.append(tier_indices)
            indexed_prefs.append(person_prefs)
        return indexed_prefs

    def _create_rankings(self, preferences: List[List[List[int]]]) -> List[List[int]]:
        """
        Create ranking matrices from preference lists with ties.

        Algorithm:
        1. Initialize rankings with n+1 (unranked).
        2. For each tier in preferences, assign ranks starting from 1.

        Example:
        For preferences [[0, 1], [2]], the rankings would be:
        - For person 0: [1, 1, 2] (0 and 1 are tied for rank 1, 2 is rank 2)

        Args:
            preferences: Indexed preference lists with ties

        Returns:
            Matrix where rankings[i][j] gives person i's rank for person j
        """
        rankings: List[List[int]] = []
        for person_prefs in preferences:
            ranking = [self.n + 1] * self.n  # Unranked partners get rank n+1
            current_rank = 1

            for tier in person_prefs:
                for target in tier:
                    ranking[target] = current_rank
                current_rank += len(tier)

            rankings.append(ranking)
        return rankings

    def _gale_shapley(
        self,
        proposer_prefs: List[List[List[int]]],
        proposer_rankings: List[List[int]],
        receiver_rankings: List[List[int]],
    ) -> List[Tuple[int, int]]:
        """
        Core Gale-Shapley algorithm implementation.

        Algorithm:
        1. Initialize free proposers and their current partners.
        2. While there are free proposers:
           - Get the next receiver to propose to.
           - If mutually acceptable, either match or replace current partner.
           - If receiver is free, match them; otherwise, check preferences.

        Args:
            proposer_prefs: Indexed preference lists for proposers
            proposer_rankings: Ranking matrix for proposers
            receiver_rankings: Ranking matrix for receivers

        Returns:
            List of tuples representing stable matchings (proposer_idx, receiver_idx)
        """
        free_proposers = list(range(self.n))
        current_partners: List[Optional[int]] = [None] * self.n  # receivers' current partners
        next_proposal_idx = [0] * self.n  # track next proposal for each proposer
        tier_position = [0] * self.n  # track position within current tier for each proposer

        while free_proposers:
            proposer = free_proposers[0]

            # Get next receiver to propose to
            receiver = self._get_next_receiver(
                proposer, next_proposal_idx, tier_position, proposer_prefs
            )

            if receiver is None:  # Proposer has exhausted preferences
                free_proposers.remove(proposer)
                continue

            # Advance to next receiver in current tier or next tier
            self._advance_proposal_position(
                proposer, next_proposal_idx, tier_position, proposer_prefs
            )

            # Check if both find each other acceptable
            if not self._mutually_acceptable(
                proposer, receiver, proposer_rankings, receiver_rankings
            ):
                continue

            # If receiver is free, match them
            if current_partners[receiver] is None:
                current_partners[receiver] = proposer
                free_proposers.remove(proposer)
            else:
                # Check if receiver prefers new proposer
                current_proposer = current_partners[receiver]
                assert current_proposer is not None, "current_proposer should not be None"
                if self._prefers_new_proposer(
                    receiver, proposer, current_proposer, receiver_rankings
                ):
                    current_partners[receiver] = proposer
                    free_proposers.remove(proposer)
                    free_proposers.append(current_proposer)

        # Create final matching list
        matches: List[Tuple[int, int]] = []
        for receiver_idx, proposer_idx in enumerate(current_partners):
            if proposer_idx is not None and proposer_rankings[proposer_idx][receiver_idx] <= self.n:
                matches.append((proposer_idx, receiver_idx))

        return matches

    def _get_next_receiver(
        self,
        proposer: int,
        next_proposal_idx: List[int],
        tier_position: List[int],
        proposer_prefs: List[List[List[int]]],
    ) -> Optional[int]:
        """
        Get the next receiver for a proposer to propose to.

        Algorithm:
        1. Check if the proposer has more tiers to propose to.
        2. If so, get the current tier and position within that tier.
        3. Return the next receiver from the current tier based on the position.

        Args:
            proposer: Index of the proposer
            next_proposal_idx: List tracking the next tier index for each proposer
            tier_position: List tracking the position within the current tier for each proposer
            proposer_prefs: Indexed preference lists for proposers

        Returns:
            The index of the next receiver to propose to, or None if no more tiers are available
        """
        if next_proposal_idx[proposer] < len(proposer_prefs[proposer]):
            current_tier = proposer_prefs[proposer][next_proposal_idx[proposer]]
            if tier_position[proposer] < len(current_tier):
                return current_tier[tier_position[proposer]]
        return None

    def _advance_proposal_position(
        self,
        proposer: int,
        next_proposal_idx: List[int],
        tier_position: List[int],
        proposer_prefs: List[List[List[int]]],
    ) -> None:
        """
        Advance the proposal position for a proposer.

        Algorithm:
        1. Check if the proposer has more tiers to propose to.
        2. If so, increment the position within the current tier.
        3. If the current tier is exhausted, move to the next tier and reset position.

        Args:
            proposer: Index of the proposer
            next_proposal_idx: List tracking the next tier index for each proposer
            tier_position: List tracking the position within the current tier for each proposer
            proposer_prefs: Indexed preference lists for proposers

        Returns:
            None
        """
        if next_proposal_idx[proposer] < len(proposer_prefs[proposer]):
            current_tier = proposer_prefs[proposer][next_proposal_idx[proposer]]
            tier_position[proposer] += 1
            # If we've exhausted current tier, move to next tier
            if tier_position[proposer] >= len(current_tier):
                next_proposal_idx[proposer] += 1
                tier_position[proposer] = 0

    def _mutually_acceptable(
        self,
        proposer: int,
        receiver: int,
        proposer_rankings: List[List[int]],
        receiver_rankings: List[List[int]],
    ) -> bool:
        """
        Check if proposer and receiver find each other acceptable.

        Algorithm:
        1. Check if the proposer ranks the receiver within their acceptable range.
        2. Check if the receiver ranks the proposer within their acceptable range.

        Args:
            proposer: Index of the proposer
            receiver: Index of the receiver
            proposer_rankings: Ranking matrix for proposers
            receiver_rankings: Ranking matrix for receivers

        Returns:
            True if both find each other acceptable, False otherwise
        """
        proposer_acceptable = proposer_rankings[proposer][receiver] <= self.n
        receiver_acceptable = receiver_rankings[receiver][proposer] <= self.n
        return proposer_acceptable and receiver_acceptable

    def _prefers_new_proposer(
        self,
        receiver: int,
        new_proposer: int,
        current_proposer: int,
        receiver_rankings: List[List[int]],
    ) -> bool:
        """
        Check if receiver prefers new proposer over current partner.

        Algorithm:
        1. Compare the rankings of the new proposer and current proposer for the receiver.
        2. Return True if the new proposer is preferred, False otherwise.

        Args:
            receiver: Index of the receiver
            new_proposer: Index of the new proposer
            current_proposer: Index of the current proposer
            receiver_rankings: Ranking matrix for receivers

        Returns:
            True if the receiver prefers the new proposer, False otherwise
        """
        new_proposer_rank = receiver_rankings[receiver][new_proposer]
        current_proposer_rank = receiver_rankings[receiver][current_proposer]
        return new_proposer_rank < current_proposer_rank


def load_preferences_from_file(
    filename: str,
) -> Tuple[List[str], List[str], Dict[str, List[List[str]]], Dict[str, List[List[str]]]]:
    """
    Load preferences from a JSON file.

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

    Args:
        filename: Path to the JSON file containing preferences

    Returns:
        Tuple of (proposers, receivers, proposer_preferences, receiver_preferences)

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

    # Validate required keys
    required_keys = ["proposers", "receivers", "proposer_preferences", "receiver_preferences"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Missing required keys in preferences file: {missing_keys}")

    proposers: List[str] = data["proposers"]
    receivers: List[str] = data["receivers"]
    proposer_preferences: Dict[str, List[List[str]]] = data["proposer_preferences"]
    receiver_preferences: Dict[str, List[List[str]]] = data["receiver_preferences"]

    # Validate preference structure
    for person, prefs in proposer_preferences.items():
        for _, tier in enumerate(prefs):
            for choice in tier:
                if choice not in receivers:
                    raise ValueError(f"Invalid receiver '{choice}' in preferences for '{person}'")

    for person, prefs in receiver_preferences.items():
        for _, tier in enumerate(prefs):
            for choice in tier:
                if choice not in proposers:
                    raise ValueError(f"Invalid proposer '{choice}' in preferences for '{person}'")

    return proposers, receivers, proposer_preferences, receiver_preferences


def display_results(result: List[Tuple[str, str]], rfs: List[str], projects: List[str]) -> None:
    """
    Display the matching results and unmatched participants.

    Args:
        result: List of (RF, Project) tuples
        rfs: List of all RFs
        projects: List of all projects

    Returns:
        None
    """
    print("Stable matching (RF -> Project):")
    for rf, project in sorted(result):
        print(f"  {rf} -> {project}")

    # Show unmatched participants
    matched_rfs = {rf for rf, _ in result}
    matched_projects = {project for _, project in result}

    unmatched_rfs = [rf for rf in rfs if rf not in matched_rfs]
    unmatched_projects = [project for project in projects if project not in matched_projects]

    if unmatched_rfs:
        print(f"\nUnmatched RFs: {', '.join(unmatched_rfs)}")
    if unmatched_projects:
        print(f"Unmatched Projects: {', '.join(unmatched_projects)}")

    print(f"\nMatching Summary: {len(result)} matches found")


def main():
    """
    RF-Project matching with file-based preferences.
    """
    preferences_file = "preferences.json"
    rfs = None
    projects = None
    rf_preferences = None
    project_preferences = None

    # Create sample file if it doesn't exist
    try:
        rfs, projects, rf_preferences, project_preferences = load_preferences_from_file(
            preferences_file
        )
        print(f"Loaded preferences from {preferences_file}")
    except FileNotFoundError:
        print(f"Preferences file '{preferences_file}' not found.")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error loading preferences: {e}")
        return

    # Create matcher and solve
    try:
        if not rfs or not projects or not rf_preferences or not project_preferences:
            raise ValueError("Preferences data is incomplete or invalid")
        matcher = StableMatchingWithTies(rfs, projects)
        result = matcher.solve(rf_preferences, project_preferences)
        display_results(result, rfs, projects)
    except (ValueError, AssertionError) as e:
        print(f"Error during matching: {e}")


if __name__ == "__main__":
    main()
