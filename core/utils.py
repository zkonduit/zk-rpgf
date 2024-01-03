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
        self.min_ratio = self.min_amount / self.total_amount

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
        df = df.sort_values(by=['project_id', 'amount'])
        print(df.dtypes)
        # get the number of votes and median amount for each project
        # convert voter address to integer IDs
        df["voter_address"] = df["voter_address"].astype("category").cat.codes
        df['project_id'] = df['project_id'].astype('category').cat.codes

        print(df.head())
        print(df.tail())
        num_projects = len(df['project_id'].unique())

        # now convert to a tensor of shape (n_projects, n_votes, 2) where the first column are the votes and the second column are the amounts
        tensor_project_allocation = torch.tensor(df[['project_id', 'voter_address', 'amount']].values.tolist(), dtype=torch.float64, requires_grad=True)
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
            values = project_tensor[:, 2:3]
            # set to all to int
            votes = values.type(torch.int64).reshape(-1)
            tensors.append(votes)
        return tensors
    
    def inner_loop(self, *x, mask=None):
        median_amounts = []
        scaled_min_amounts = []
        votes = []
        for tensor in x:
            num_bids = tensor.shape[0]
            votes_count = torch.tensor([num_bids]).reshape(1, 1)
            median_amount_ceil = torch.topk(tensor, k=num_bids // 2 + 1).values[-1].reshape(1, 1)
            median_amount_floor = torch.topk(tensor, k=num_bids // 2 + 1, largest = False).values[-1].reshape(1, 1)
            # # calculate the median amount
            median_amount = (median_amount_ceil + median_amount_floor) / 2
            # concatenate the results
            median_amounts.append(median_amount)
            # we keep a running counter of the scaled min amount to prevent having an extremely large sum later on (overflows the extended K capacity of halo2 with SRS with k=26 (max public) when inverted using a lookup table)
            scaled_min_amounts.append(median_amount_ceil * self.min_ratio)
            votes.append(votes_count)

        votes = torch.cat(votes, dim=0)
         # is eligible
        is_eligible = votes >= self.quorum

        median_amounts = torch.cat(median_amounts, dim=0)

        if mask is not None:
            median_amounts = median_amounts * mask
        # sum to get the total scaled min amount
        scaled_min_amounts = torch.cat(scaled_min_amounts, dim=0)

        if mask is not None:
            scaled_min_amounts = scaled_min_amounts * mask

         # now scale the allocations to the total amount of OP and filter out those with less than 1500 OP
        #  scaled min is equal to sum(scaled_min_amounts) = sum(median_amounts) * min_ratio = sum(median_amounts) * min_amount / total_amount
        scaled_min_sum = torch.sum(scaled_min_amounts)

        # meets minimum amount
        meets_min = median_amounts >= scaled_min_sum

        # note that this is equivalent to median_amounts / scaled_min = median_amounts * total_amount / (sum(median_amounts) * min_amount)
        scaled_by_min_median = median_amounts / torch.sum(scaled_min_amounts * is_eligible) 
        rescaled_median_amounts = self.min_amount * scaled_by_min_median * meets_min * is_eligible

        project_allocation = torch.cat([votes, median_amounts, is_eligible, rescaled_median_amounts], dim=1)
        return (project_allocation, meets_min)


    def forward(self, *x, num_iterations=1):
        """
        Calculate the raw allocation amount of each project, then scale allocations to the total amount of OP and filter out those with less than 1500 OP.
        """

        project_allocation_first, meets_min = self.inner_loop(*x)

        # we've just calculated the first iteration
        for i in range(num_iterations - 1):
            project_allocation_iter, meets_min = self.inner_loop(*x, mask=meets_min)
        if num_iterations == 1:
            project_allocation = project_allocation_first
        else:
            # concat columns 0-2 from the first iteration with the last column from the last iteration
            project_allocation = torch.cat([project_allocation_first[:, 0:3], project_allocation_iter[:, 3:4]], dim=1)
        return project_allocation

