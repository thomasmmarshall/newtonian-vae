

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import itertools
from pprint import pprint

from dm_control import suite
from control.dm_control import multireacher, linearmass, panda

imported_envs = [("multireacher", multireacher),
                 ("linearmass", linearmass),
                 ("panda", panda)]

suite._DOMAINS.update({name: module for name, module in imported_envs
                        if inspect.ismodule(module) and hasattr(module, 'SUITE')})

suite.ALL_TASKS = suite._get_tasks(tag=None)

# Subsets of ALL_TASKS, generated via the tag mechanism.
suite.BENCHMARKING = suite._get_tasks('benchmarking')
suite.EASY = suite._get_tasks('easy') 
suite.HARD = suite._get_tasks('hard')
suite.EXTRA = tuple(sorted(set(suite.ALL_TASKS) - set(suite.BENCHMARKING)))

# A mapping from each domain name to a sequence of its task names.
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)