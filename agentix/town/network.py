"""Static facts about the NOMINAL network (the utility's own model): junctions, distances, zones, demand, coordinates.

This is public knowledge: every method may use it. The hidden truth model (official L-TOWN_Real.inp) lives in
`agentix.sim` and is never imported by a method.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property

import networkx as nx
import numpy as np
import wntr


def load_model(inp_path: str) -> wntr.network.WaterNetworkModel:
    return wntr.network.WaterNetworkModel(inp_path)


@dataclass
class NetworkInfo:
    wn: wntr.network.WaterNetworkModel
    name: str = "ltown"
    cache_dir: str | None = None                   # where the all-pairs pipe distance is cached
    _extra: dict = field(default_factory=dict, repr=False)

    @cached_property
    def junctions(self) -> list[str]:
        return list(self.wn.junction_name_list)

    @cached_property
    def index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.junctions)}

    @cached_property
    def elevation(self) -> np.ndarray:
        return np.array([self.wn.get_node(n).elevation for n in self.junctions])

    @cached_property
    def xy(self) -> np.ndarray:
        return np.array([self.wn.get_node(n).coordinates for n in self.junctions], dtype=float)

    @cached_property
    def base_demand(self) -> np.ndarray:
        """Mean base demand per junction (m3/s), summed over demand categories."""
        return np.array([sum(d.base_value for d in self.wn.get_node(n).demand_timeseries_list)
                         for n in self.junctions])

    @cached_property
    def graph(self) -> nx.Graph:
        """Undirected graph of all links weighted by length (valves and pumps get length 1)."""
        g = nx.Graph()
        for name, link in self.wn.links():
            g.add_edge(link.start_node_name, link.end_node_name,
                       length=float(getattr(link, "length", 1.0) or 1.0), link=name)
        return g

    @cached_property
    def pipe_graph(self) -> nx.Graph:
        """Pipes only: its connected components are the pressure zones."""
        g = nx.Graph()
        for name, link in self.wn.links():
            if link.link_type == "Pipe":
                g.add_edge(link.start_node_name, link.end_node_name, length=float(link.length), link=name)
        return g

    @cached_property
    def zone_of(self) -> dict[str, str]:
        """Zone per junction: the largest pipe-connected area is A; the others B (low) and C (high, pumped)."""
        comps = [c for c in nx.connected_components(self.pipe_graph) if sum(n in self.index for n in c) > 1]
        comps.sort(key=len, reverse=True)
        rest = sorted(comps[1:], key=lambda c: -np.mean([self.wn.get_node(n).elevation for n in c if n in self.index]))
        names = ["A"] + [chr(ord("C") - i) for i in range(len(rest))]
        labels = {}
        for lab, comp in zip(names, [comps[0]] + rest):
            labels.update({n: lab for n in comp if n in self.index})
        return {n: labels.get(n, "A") for n in self.junctions}

    @cached_property
    def pipe_distance(self) -> np.ndarray:
        """All-pairs shortest distance along the pipes (m), [J, J]. Cached on disk (takes ~20 s to compute)."""
        path = os.path.join(self.cache_dir, "pipe_distance.npy") if self.cache_dir else None
        if path and os.path.exists(path):
            return np.load(path)
        out = np.full((len(self.junctions), len(self.junctions)), np.inf, dtype=np.float32)
        for i, s in enumerate(self.junctions):
            for n, v in nx.single_source_dijkstra_path_length(self.graph, s, weight="length").items():
                j = self.index.get(n)
                if j is not None:
                    out[i, j] = v
        if path:
            np.save(path, out)
        return out

    @cached_property
    def pipe_length_at_node(self) -> np.ndarray:
        """Half the length of every pipe attached to a junction (m): leaks happen per km of pipe."""
        L = np.zeros(len(self.junctions))
        for u, v, d in self.pipe_graph.edges(data=True):
            for n in (u, v):
                if n in self.index:
                    L[self.index[n]] += d["length"] / 2
        return L

    def pipes(self) -> list[str]:
        return list(self.wn.pipe_name_list)

    def pipe_endpoints(self, pipe: str) -> tuple[str, str]:
        link = self.wn.get_link(pipe)
        return link.start_node_name, link.end_node_name


def series_chain(wn, pipe: str) -> list[str]:
    """The pipes in series with `pipe` (joined only through junctions with exactly two pipes): one stretch of main (public map)."""
    if pipe not in wn.pipe_name_list:
        return [pipe]
    out, todo = {pipe}, [pipe]
    while todo:
        link = wn.get_link(todo.pop())
        for n in (link.start_node_name, link.end_node_name):
            if n in wn.junction_name_list:
                links = wn.get_links_for_node(n)                      # every link: a valve or pump at a junction ends the stretch
                if len(links) == 2 and all(l in wn.pipe_name_list for l in links):
                    for l in links:
                        if l not in out:
                            out.add(l)
                            todo.append(l)
    return sorted(out)
