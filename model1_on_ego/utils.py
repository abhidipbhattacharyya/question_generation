import os
import os.path as op
import errno

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def csv_writer(values, csv_file_name, sep=','):
    mkdir(os.path.dirname(csv_file_name))
    csv_file_name_tmp = csv_file_name + '.tmp'
    with open(csv_file_name_tmp, 'wb') as fp:
        assert values is not None
        for value in values:
            assert value is not None
            v = sep.join(map(lambda v: v.decode() if type(v) == bytes else str(v), value)) + '\n'
            v = v.encode()
            fp.write(v)
    os.rename(csv_file_name_tmp, csv_file_name)


def concat_csv_files(cache_files, predict_file):
    all_lines = []
    with open(predict_file,'w') as fout:
        for cf in cache_files:
            with open(cf,'r') as fin:
                lines = fin.readlines()
                for l in lines:
                    fout.write(l.strip()+"\n")


def delete_csv_files(csvs):
    for c in csvs:
        if op.isfile(c):
            os.remove(c)

def reorder_csv_keys(i_file, pair_ids, o_file):
    with open(i_file, "r") as f:
        lines = f.readlines()

    line_dic = {}
    for l in lines:
        linfo = l.strip().split(",")
        p_id = linfo[0]
        #op = l.strip()
        line_dic[p_id] = l.strip()

    with open(o_file, "w") as f:
        for pid in pair_ids:
            f.write(line_dic[pid]+"\n")
