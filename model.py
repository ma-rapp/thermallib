import json
import numpy as np
import os
import struct
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MATEX_PATH = os.path.join(HERE, 'MatEx')
MATEX_BIN = os.path.join(MATEX_PATH, 'MatEx')
MATEX_CONFIG = os.path.join(MATEX_PATH, 'matex.config')
HOTSPOT_PATH = os.path.join(HERE, 'HotSpot-6.0')
HOTSPOT_BIN = os.path.join(HOTSPOT_PATH, 'hotspot')
HOTSPOT_CONFIG = os.path.join(HOTSPOT_PATH, 'hotspot.config')
GRID_THERMAL_MAP_BIN = os.path.join(HOTSPOT_PATH, 'grid_thermal_map.pl')
GRID_THERMAL_MAP_PURE_BIN = os.path.join(HERE, 'grid_thermal_map_pure.pl')

FLOORPLAN = 'floorplan.flp'
POWERTRACE = 'powertrace.ptrace'
EIGENDATA = 'eigendata.bin'
MODELINFO = 'modelinfo.dat'

CORE_ASPECT_RATIO = 1  # width / height

TDP_TEMPERATURE_ACCURACY = 0.1


class ThermalModel(object):
    def __init__(self, folder, unitNames, BInv, G, eigenValues, eigenVectors, eigenVectorsInv):
        self.folder = folder
        self.unitNames = unitNames
        self.BInv = BInv
        self.G = G
        self.eigenValues = eigenValues
        self.eigenVectors = eigenVectors
        self.eigenVectorsInv = eigenVectorsInv

    @property
    def floorplan_filename(self):
        return os.path.join(self.folder, FLOORPLAN)


def _get_core_id(processorsY, processorsX, x, y, core_naming):
    assert core_naming in ('xy', 'sniper')
    assert 0 <= y < processorsY
    assert 0 <= x < processorsX
    if core_naming == 'sniper':
        return 'Core{:d}-TP'.format(y * processorsX + x)
    else:
        assert core_naming == 'xy'
        return 'Core_{:d},{:d}'.format(y + 1, x + 1)


def _generate_floorplan(folder, processorsY, processorsX, core_width, core_naming):
    if not os.path.exists(folder):
        os.mkdir(folder)

    filename = os.path.join(folder, FLOORPLAN)
    core_height = core_width / CORE_ASPECT_RATIO

    core_width = round(core_width, 6)
    core_height = round(core_height, 6)

    with open(filename, 'w') as f:
        f.write('# Line Format: <unit-name>\\t<width>\\t<height>\\t<left-x>\\t<bottom-y>\n')
        for y in range(processorsY):
            for x in range(processorsX):
                coreID = _get_core_id(processorsY, processorsX, x, y, core_naming)
                left = x * core_width
                bottom = (processorsY - y - 1) * core_height
                f.write('{}\t{:f}\t{:f}\t{:f}\t{:f}\n'.format(coreID, core_width, core_height, left, bottom))


def _get_core_dimensions(folder):
    filename = os.path.join(folder, FLOORPLAN)
    dimensions = []
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            cols = line.split('\t')
            dimensions.append((float(cols[1]), float(cols[2])))
    return dimensions


def _generate_power_trace(folder_or_filename, processorsY, processorsX, core_naming, include_time=False, per_core_power=None):
    if folder_or_filename.endswith('.ptrace'):
        filename = folder_or_filename
    else:
        filename = os.path.join(folder_or_filename, POWERTRACE)
    with open(filename, 'w') as f:
        if include_time:
            f.write('time\t')
        for y in range(processorsY):
            for x in range(processorsX):
                if (x, y) != (0, 0):
                    f.write('\t')
                coreID = _get_core_id(processorsY, processorsX, x, y, core_naming)
                f.write('{}'.format(coreID))
        f.write('\n')
        if per_core_power:
            if include_time:
                f.write('0\t')
            for y in range(processorsY):
                for x in range(processorsX):
                    if (x, y) != (0, 0):
                        f.write('\t')
                    f.write('{}'.format(per_core_power))
            f.write('\n')


def _get_peak_temperature_at_tdp(folder, processorsY, processorsX, tdp, ambient_temperature, core_naming, config):
    if config is None:
        config = {}
    per_core_power = tdp / (processorsX * processorsY)
    floorplan_filename = os.path.join(folder, FLOORPLAN)
    temp_power_filename = os.path.join(folder, 'tdpcheck.ptrace')
    temp_steady_filename = os.path.join(folder, 'tdpcheck.steady')
    _generate_power_trace(temp_power_filename, processorsY, processorsX, core_naming, include_time=False, per_core_power=per_core_power)

    params = []
    for key, value in config.items():
        params += ['-' + str(key), str(value)]
    subprocess.check_output([HOTSPOT_BIN,
                             '-c', HOTSPOT_CONFIG,
                             '-f', floorplan_filename,
                             '-p', temp_power_filename,
                             '-steady_file', temp_steady_filename,
                             '-ambient', str(ambient_temperature)]
                            + params)

    steady_temperatures = []
    with open(temp_steady_filename, 'r') as f:
        for line in f:
            core_id, temperature = line.strip().split()
            steady_temperatures.append(float(temperature))

    return max(steady_temperatures)

def _generate_eigen_bin(folder, config):
    if config is None:
        config = {}

    floorplan_filename = os.path.join(folder, FLOORPLAN)
    powertrace_filename = os.path.join(folder, POWERTRACE)
    eigen_filename = os.path.join(folder, EIGENDATA)
    params = []
    for key, value in config.items():
        params += ['-' + str(key), str(value)]
    subprocess.check_output([MATEX_BIN,
                             '-c', MATEX_CONFIG,
                             '-f', floorplan_filename,
                             '-p', powertrace_filename,
                             '-eigen_out', eigen_filename]
                            + params)


def _create_and_return_error(folder, processorsY, processorsX, core_width, tdp, ambient_temperature, max_temperature, core_naming='xy', config=None):
    if config is None:
        config = {}

    _generate_floorplan(folder, processorsY, processorsX, core_width, core_naming)
    peak_temperature = _get_peak_temperature_at_tdp(folder, processorsY, processorsX, tdp, ambient_temperature, core_naming, config)
    error = max_temperature - peak_temperature
    if abs(error) < TDP_TEMPERATURE_ACCURACY:
        _generate_power_trace(folder, processorsY, processorsX, core_naming, include_time=True)
        _generate_eigen_bin(folder, config)
        return 0
    else:
        return error


def _get_modelinfo(folder, processorsY, processorsX, tdp, ambient_temperature, max_temperature, core_naming='sniper', config=None):
    return {'processorsX': processorsX,
            'processorsY': processorsY,
            'tdp': tdp,
            'ambient_temperature': ambient_temperature,
            'max_temperature': max_temperature,
            'core_naming': core_naming,
            'config': config}


def load_modelinfo(folder):
    with open(os.path.join(folder, MODELINFO), 'r') as f:
        return json.load(f)


def _up_to_date(folder, processorsY, processorsX, tdp, ambient_temperature, max_temperature, core_naming='sniper', config=None):
    if not all(os.path.exists(os.path.join(folder, filename)) for filename in (FLOORPLAN, POWERTRACE, EIGENDATA, MODELINFO)):
        return False

    modelinfo = _get_modelinfo(folder, processorsY, processorsX, tdp, ambient_temperature, max_temperature, core_naming, config)
    if load_modelinfo(folder) != modelinfo:
        return False

    peak_temperature = _get_peak_temperature_at_tdp(folder, processorsY, processorsX, tdp, ambient_temperature, core_naming, config)
    if abs(peak_temperature - max_temperature) > TDP_TEMPERATURE_ACCURACY:
        return False
    
    return True


def create(folder, processorsX, processorsY, tdp, ambient_temperature, max_temperature, core_naming='sniper', config=None):
    """
    create a floorplan and thermal model in the given folder
    max_temperature is reached with the accuracy of 1C if the chip is operated at TDP.

    core_naming is either 'xy' or 'sniper'
    """
    if _up_to_date(folder, processorsX, processorsY, tdp, ambient_temperature, max_temperature, core_naming, config):
        return

    next_test_size = 1.8 / 1000
    known_lower_bound = None
    known_upper_bound = None
    for i in range(1000):
        print('trying per-core size: {:.2f} mm'.format(next_test_size * 1000))
        error = _create_and_return_error(folder, processorsX, processorsY, next_test_size, tdp, ambient_temperature, max_temperature, core_naming, config)
        if error == 0:
            modelinfo = _get_modelinfo(folder, processorsX, processorsY, tdp, ambient_temperature, max_temperature, core_naming, config)
            with open(os.path.join(folder, MODELINFO), 'w') as f:
                json.dump(modelinfo, f, indent=4, sort_keys=True)
            return
        elif error < 0:
            known_lower_bound = (next_test_size, error)
        else:
            known_upper_bound = (next_test_size, error)
    
        if known_lower_bound is None:
            next_test_size /= 2
        elif known_upper_bound is None:
            next_test_size *= 2
        else:
            s1, e1 = known_lower_bound
            s2, e2 = known_upper_bound
            next_test_size = s1 - (s2 - s1) / (e2 - e1) * e1

    raise Exception('max. iterations reached')


def create_fixed_size(folder, processorsX, processorsY, core_width, ambient_temperature, max_temperature, core_naming='sniper', config=None):
    """
    create a floorplan and thermal model in the given folder

    core_naming is either 'xy' or 'sniper'
    """
    if all(os.path.exists(os.path.join(folder, filename)) for filename in (FLOORPLAN, POWERTRACE, EIGENDATA, MODELINFO)):
        if abs(_get_core_dimensions(folder)[0][0] - core_width) < 1e-6:
            return

    _generate_floorplan(folder, processorsY, processorsX, core_width, core_naming)
    _generate_power_trace(folder, processorsY, processorsX, core_naming, include_time=True)
    _generate_eigen_bin(folder, config)
    tdp = None
    modelinfo = _get_modelinfo(folder, processorsX, processorsY, tdp, ambient_temperature, max_temperature, core_naming, config)
    with open(os.path.join(folder, MODELINFO), 'w') as f:
        json.dump(modelinfo, f, indent=4, sort_keys=True)


def _read_int(f):
    data = f.read(4)
    return struct.unpack('i', data)[0]

def _read_float(f):
    data = f.read(4)
    return struct.unpack('f', data)[0]

def _read_double(f):
    data = f.read(8)
    return struct.unpack('d', data)[0]

def _read_double_matrix(f, rows, columns):
    matrix = []
    for r in range(rows):
        row = []
        for c in range(columns):
            row.append(_read_double(f))
        matrix.append(row)
    return np.asarray(matrix)

def _read_double_vector(f, elements):
    vector = []
    for n in range(elements):
        vector.append(_read_double(f))
    return np.asarray(vector)

def _read_line(f):
    data = ""
    c = f.read(1)
    while c not in (b"", b"\n"):
        data += chr(c[0]) if isinstance(c[0], int) else c[0]
        c = f.read(1)
    return data

def load(folder):
    """
    load the thermal model from the given folder
    """
    floorplan_filename = os.path.join(folder, FLOORPLAN)
    powertrace_filename = os.path.join(folder, POWERTRACE)
    eigen_filename = os.path.join(folder, EIGENDATA)

    with open(eigen_filename, 'rb') as f:
        numberUnits = _read_int(f)
        numberNodesAmbient = _read_int(f)
        numberThermalNodes = _read_int(f)
        assert numberThermalNodes == 4 * numberUnits + 12
        assert numberNodesAmbient == numberThermalNodes - 3 * numberUnits

        unitNames = []
        for u in range(numberUnits):
            unitName = _read_line(f)
            unitNames.append(unitName)
            #width = readDouble(f)
            #height = readDouble(f)

        BInv = _read_double_matrix(f, numberThermalNodes, numberThermalNodes)

        G = [0.0] * (numberThermalNodes - numberNodesAmbient)
        for n in range(numberNodesAmbient):
            G.append(_read_double(f))
        G = np.asarray(G)

        eigenValues = _read_double_vector(f, numberThermalNodes)
        eigenVectors = _read_double_matrix(f, numberThermalNodes, numberThermalNodes)
        eigenVectorsInv = _read_double_matrix(f, numberThermalNodes, numberThermalNodes)

        assert f.read(1) == b''

    return ThermalModel(folder, unitNames, BInv, G, eigenValues, eigenVectors, eigenVectorsInv)
