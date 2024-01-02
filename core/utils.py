# -*- coding: utf-8 -*-
import logging
import pandas as pd
import json
import torch
import torch.nn as nn

from typing import List


def get_logger() -> logging.Logger:
    """
    Get logger instance.
    """
    level = logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def safe_json_loads(s):
    """
    Safely loads a JSON string, replacing single quotes with double quotes, and returns a JSON string.
    """
    try:
        # Convert to valid JSON format and parse
        parsed = json.loads(s.replace("'", '"'))
        return json.dumps(parsed)  # Convert back to JSON string
    except json.JSONDecodeError:
        # Handle or log the error if needed
        return json.dumps([])  # Return an empty JSON array as a string


def expand_json(row, idx):
    """
    Normalize and Expand the JSON-like strings or already parsed JSON.
    """
    if isinstance(row, str):
        parsed_row = json.loads(row)
    elif isinstance(row, list):
        parsed_row = row
    else:
        parsed_row = []

    expanded = pd.json_normalize(parsed_row)
    expanded["original_index"] = idx
    return expanded


class ProjectAllocator(nn.Module):
    def __init__(self, total_amount, min_amount, quorum) -> None:
        """
        Initialize the ProjectAllocator.
        """
        super(ProjectAllocator, self).__init__()
        self.total_amount = total_amount
        self.min_amount = min_amount
        self.quorum = quorum

    def calculate_initial_allocation(self, df) -> pd.DataFrame:
        """
        Calculate the raw allocation amount of each project.
        """
        # get the number of votes and median amount for each project
        project_allocation = df.groupby("project_id").agg(
            votes_count=("voter_address", "count"), median_amount=("amount", "median")
        )

        # if the number of votes is less than the quorum, the project is not eligible
        project_allocation["is_eligible"] = (
            project_allocation["votes_count"] >= self.quorum
        )

        return project_allocation.sort_values("median_amount", ascending=False)

    def convert_df_to_tensor(self, df) -> torch.Tensor:
        """
        Convert the dataframe to a tensor of shape (num_votes, 2) where the first column are the votes and the second column are the amounts
        """
        # get the number of votes and median amount for each project
        # convert voter address to integer IDs
        df["voter_address"] = df["voter_address"].astype("category").cat.codes
        df['project_id'] = df['project_id'].astype('category').cat.codes
        num_projects = len(df['project_id'].unique())

        # now convert to a tensor of shape (n_projects, n_votes, 2) where the first column are the votes and the second column are the amounts
        tensor_project_allocation = torch.tensor(df[['project_id', 'voter_address', 'amount']].values.tolist(), dtype=torch.float32, requires_grad=True)
        return tensor_project_allocation, num_projects


    # scaling the total to RPGF OP total by project and filter out those with < min OP requirement
    def scale_allocations(self, df) -> pd.DataFrame:
        """
        Scale the allocations to the total amount of OP and filter out those with less than 1500 OP.
        """
        scale_factor = self.total_amount / df["median_amount"].sum()
        df["scaled_amount"] = df["median_amount"] * scale_factor

        df = df[df["scaled_amount"] >= self.min_amount]

        return df

    def scale_allocations_oneby(self, df) -> pd.DataFrame:
        """
        Scale the allocations to the total amount of OP and filter out those with less than 1500 OP.
        """
        log = get_logger()

        amount_eligible = df["median_amount"].sum()
        scale_factor = self.total_amount / amount_eligible

        log.info("Check - Original Amount Eligible: " + str(amount_eligible))
        log.info("Check - Scale Factor: " + str(scale_factor))

        df["scaled_amount"] = df["median_amount"] * scale_factor

        to_cut = (
            df[df["scaled_amount"] < self.min_amount]
            .sort_values(by="scaled_amount")
            .head(1)
        )

        # Print the project_id of the project to cut
        # Since project_id is the index, use index[0] to access it
        if to_cut.empty:
            log.info("Check - No projects below minimum OP")
        else:
            log.info("Check - Project cut below minimum OP: " + str(to_cut.index[0]))
            df = df[~df.index.isin(to_cut.index)]

        return df

    def get_project_tensor(self, tensor, num_projects):
        tensors = []
        for project_id in range(0, num_projects):
            project_tensor = tensor[tensor[:, 0] == project_id]
            tensors.append(project_tensor[:, 1:3])
        return tensors


    def forward(self, *x):
        """
        Calculate the raw allocation amount of each project, then scale allocations to the total amount of OP and filter out those with less than 1500 OP.
        """
        res = []
        for tensor in x:
            count = torch.unique(tensor[:, 0]).shape[0]
            votes_count = torch.tensor([count]).reshape(1, 1)
            num_bids = tensor.shape[0]
            median_amount = torch.topk(tensor[:, 1], k=num_bids // 2 + 1).values[-1].reshape(1, 1)
            # concatenate the results
            res.append(torch.cat([votes_count, median_amount], dim=1))
        project_allocation = torch.cat(res, dim=0)
        sum_median = torch.sum(project_allocation[:, 1])

        # now scale the allocations to the total amount of OP and filter out those with less than 1500 OP
        scale_factor = self.total_amount / sum_median
        project_allocation[:, 0] = project_allocation[:, 0] * scale_factor
        # project is elibible if it has more than the quorum and the amount is greater than the minimum
        is_eligible = torch.logical_and(project_allocation[:, 1] >= self.min_amount, project_allocation[:, 0] >= self.quorum).reshape(-1, 1)

        project_allocation = torch.cat([project_allocation, is_eligible], dim=1)
        return project_allocation
