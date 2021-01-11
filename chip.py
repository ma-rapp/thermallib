import model
import numpy as np
import os
import random
import scipy.linalg
import subprocess


class Chip(object):
    def __init__(self, model, processorsY, processorsX, ambient_temperature, max_temperature, inactive_power, config):
        self.model = model
        self.processorsY = processorsY
        self.processorsX = processorsX
        self.config = config if config is not None else {}
        self.ambient_temperature = ambient_temperature
        self.max_temperature = max_temperature
        self.inactive_power = inactive_power

    @property
    def cores(self):
        return self.processorsY * self.processorsX

    @property
    def shape(self):
        return self.processorsY, self.processorsX

    def get_steady_state(self, powers):
        if powers.shape != self.model.G.shape:
            p = np.zeros_like(self.model.G)
            p[0:powers.size] = np.asarray(powers).astype(dtype=float).ravel()
        else:
            p = powers
        steady = np.dot(self.model.BInv, p + self.ambient_temperature * self.model.G)
        return steady[:powers.size].reshape(powers.shape)

    def plot_grid_steady_state(self, powers, svg_filename, pure=False, temperature_range=None, create_pdf=False):
        """
        pure: no legend, no labels
        """
        tmp_dir = os.path.join('/tmp', str(random.randint(0, 100000)))
        os.mkdir(tmp_dir)
        try:
            power_file = os.path.join(tmp_dir, 'power.ptrace')
            grid_steady_file = os.path.join(tmp_dir, 'grid.steady')
            with open(power_file, 'w') as f:
                f.write('\t'.join(self.model.unitNames))
                f.write('\n')
                f.write('\t'.join(map(str, powers.ravel())))
                f.write('\n')
            params = []
            for key, value in self.config.items():
                params += ['-' + str(key), str(value)]
            subprocess.check_output([model.HOTSPOT_BIN,
                                    '-c', model.HOTSPOT_CONFIG,
                                    '-f', self.model.floorplan_filename,
                                    '-p', power_file,
                                    '-model_type', 'grid',
                                    '-grid_steady_file', grid_steady_file,
                                    '-ambient', str(self.ambient_temperature)] + 
                                    params)
            temperature_range_params = [] if temperature_range is None else [str(temperature_range[0]), str(temperature_range[1])]
            svg_content = subprocess.check_output([model.GRID_THERMAL_MAP_PURE_BIN if pure else model.GRID_THERMAL_MAP_BIN,
                                                   self.model.floorplan_filename,
                                                   grid_steady_file,
                                                   '64',
                                                   '64'] + temperature_range_params)
            with open(svg_filename, 'wb') as f:
                f.write(svg_content)

            if create_pdf:
                assert svg_filename[-4:] == '.svg'
                pdf_filename = svg_filename[:-4] + '.pdf'
                subprocess.check_output(['inkscape',
                                        '-z',
                                        '-D',
                                        '--file={}'.format(svg_filename),
                                        '--export-pdf={}'.format(pdf_filename)])
        finally:
            subprocess.check_call(['rm', '-rf', tmp_dir])

    def tdp(self):
        all_active = np.ones(self.shape)
        return self.cores * self.tsp_for_mapping(all_active)

    def tsp_for_mapping(self, active_cores):
        active_cores = active_cores.ravel()
        n = active_cores.size
        return np.min(((self.max_temperature - self.ambient_temperature) - self.inactive_power * np.dot(self.model.BInv[0:n,0:n], 1 - active_cores)) / np.dot(self.model.BInv[0:n,0:n], active_cores))

    def tsp_for_known_power(self, active_cores, known_power):
        active_cores = active_cores.ravel()
        n = active_cores.size
        return np.min(((self.max_temperature - self.ambient_temperature) - np.dot(self.model.BInv[0:n,0:n], known_power.ravel())) / np.dot(self.model.BInv[0:n,0:n], active_cores))

    def power_budget_max_steady_state(self, active_cores):
        inactive_cores = 1 - active_cores
        t_inactive = self.get_steady_state(inactive_cores * self.inactive_power)
        headroom = self.max_temperature - t_inactive
        active_indices = np.where(active_cores.ravel() != 0)[0]
        BInvTrunc = (self.model.BInv[active_indices,:])[:,active_indices]
        headroomTrunc = headroom.ravel()[active_indices]
        # BInvTrunc * powersTrunc = headroomTrunc

        if False:  # numpy linalg Ax=b
            powersTrunc = np.linalg.solve(BInvTrunc, headroomTrunc)
        if False: # numpy LU decomposition
            P, L, U = scipy.linalg.lu(BInvTrunc)
            assert np.all(P == np.identity(len(active_indices)))  # P is diagonal, i.e. no permutation happened

            y = np.linalg.solve(L, headroomTrunc)
            powersTrunc = np.linalg.solve(U, y)
        if True:  # gauss
            powersTrunc = self._gauss(BInvTrunc, headroomTrunc)

        powers = np.zeros(active_cores.size, dtype='float') + self.inactive_power
        powers[active_indices] = powersTrunc
        return powers.reshape(active_cores.shape)

    def _gauss(self, A, b):
        A = np.copy(A)
        b = np.copy(b)
        for row in range(b.size):
            # divide
            b[row] /= A[row,row]
            A[row,:] /= A[row,row]
            # add
            for row2 in range(b.size):
                if row == row2:
                    continue
                b[row2] -= A[row2,row] * b[row]
                A[row2,:] -= A[row2,row] * A[row,:]
        return b

    def get_amd(self, y, x):
        smd = 0
        for yi in range(self.processorsY):
            for xi in range(self.processorsX):
                md = abs(y - yi) + abs(x - xi)
                smd += md
        return smd / self.cores

    def get_amds(self):
        amds = [[self.get_amd(y, x) for x in range(self.processorsX)] for y in range(self.processorsY)]
        return np.asarray(amds)

    def greedy_mapping(self, active_cores, available, k):
        def single_core(shape, location):
            mask = np.zeros(shape)
            mask[location] = 1
            return mask
        mask = np.zeros(self.shape)
        for i in range(k):
            indices = zip(*np.nonzero(available))
            tsps = {index: self.tsp_for_mapping(active_cores + mask + single_core(mask.shape, index)) for index in indices}
            best_index = max(tsps.keys(), key=lambda index: tsps[index])
            available[best_index] = 0
            mask[best_index] = 1
        return mask

    def _all_mappings(self, available, k):
        if k == 0:
            yield np.zeros(self.shape)
        elif available.sum() < k:
            pass
        else:
            indices = list(zip(*np.nonzero(available)))
            for index in indices:
                available[index] = 0
                for mapping in self._all_mappings(available.copy(), k - 1):
                    mapping[index] = 1
                    yield mapping

    def optimal_mapping(self, active_cores, available, k):
        assert self.inactive_power == 0
        lowest_temp = 0
        best_mapping = None
        i = 0
        for mapping in self._all_mappings(available, k):
            i += 1
            temp = np.dot(self.model.BInv[:mapping.size,:mapping.size], mapping.ravel()).max()
            if best_mapping is None or temp < lowest_temp:
                lowest_temp = temp
                best_mapping = mapping
        return best_mapping

    def near_pareto_optimal_mappings(self, active_cores, k):
        amds = self.get_amds()
        mappings = {}  # highest_unique_amd -> (k new cores mapping, TSP)
        for amd in sorted(np.unique(amds)):
            available = np.logical_and(amds <= amd, active_cores == 0)
            if np.count_nonzero(available) < k:
                continue
            mask = self.greedy_mapping(active_cores, available, k)
            mappings[amd] = (mask, self.tsp_for_mapping(active_cores + mask))
        return mappings

    def PCMap(self, active_cores, k):
        candidates = self.near_pareto_optimal_mappings(active_cores, k)
        tsps = [tsp for mask, tsp in candidates.values()]
        if len(tsps) > 1:
            alpha = (max(tsps) - min(tsps)) / (max(candidates.keys()) - min(candidates.keys()))
        else:
            alpha = 1
        mask, rating = max(((mask, tsp - alpha * amd) for amd, (mask, tsp) in candidates.items()), key=lambda mr: mr[1])
        return mask

    def _format_matlab_matrix(self, name, matrix):
        lines = []
        lines.append('{} = [\n'.format(name))
        for row in matrix:
            formatted_values = ['{:f}'.format(v) for v in row]
            lines.append('    [{}]\n'.format(' '.join(formatted_values)))
        lines.append('];\n')
        return ''.join(lines)

    def create_matlab_code(self):
        with open(os.path.join(self.model.folder, 'model.m'), 'w') as f:
            f.write(self._format_matlab_matrix('BInv', self.model.BInv) + '\n')
            f.write('inactivePower = {:f};\n'.format(self.inactive_power))
            f.write('ambientTemperature = {:f};\n'.format(self.ambient_temperature))
            f.write('maxTemperature = {:f};\n'.format(self.max_temperature))

    @classmethod
    def load_or_create(cls, folder, processorsY, processorsX, tdp, ambient_temperature, max_temperature, inactive_power, core_naming='sniper', config=None):
        model.create(folder, processorsY, processorsX, tdp, ambient_temperature, max_temperature, core_naming, config)
        return cls.load(folder, inactive_power)

    @classmethod
    def load_or_create_fixed_size(cls, folder, processorsY, processorsX, core_width, ambient_temperature, max_temperature, inactive_power, core_naming='sniper', config=None):
        model.create_fixed_size(folder, processorsY, processorsX, core_width, ambient_temperature, max_temperature, core_naming, config)
        return cls.load(folder, inactive_power)

    @classmethod
    def load(cls, folder, inactive_power):
        modelinfo = model.load_modelinfo(folder)
        return cls(model.load(folder),
                   modelinfo['processorsY'],
                   modelinfo['processorsX'],
                   modelinfo['ambient_temperature'],
                   modelinfo['max_temperature'],
                   inactive_power,
                   modelinfo['config'])
