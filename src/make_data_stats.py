# DATT PROCESSING OF STATS DATASET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import time
import scipy.io.wavfile

from bs4 import BeautifulSoup
import bs4
import requests
import urllib
import subprocess

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_RAW = os.path.join(CURR_PATH, os.pardir, 'data', 'raw')
TEST = os.path.join(CURR_PATH, os.pardir, 'data', 'TEST')

#(os.sep).join(os.getcwd().split(os.sep)[:-1])+os.sep + 'data'+ os.sep + 'raw'
PROCESSED = os.path.join(CURR_PATH, os.pardir, 'data', 'processed')
STATS = os.path.join(PROCESSED, 'stats')

sys.path.append(CURR_PATH)
# CHANGE TEST WITH DATA_RAW
def download_extract(dirsave):
    from bs4 import BeautifulSoup
    import bs4
    import requests
    import urllib
    page = requests.get("http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/")
    soup = BeautifulSoup(page.content, 'html.parser')

    body = list(soup.children)[7]
    links = np.array([bodyi.get('href') for bodyi in list(body.children) if isinstance(bodyi, bs4.element.Tag)])
    links = [x for x in links[(links != None)] if x.endswith('tgz')]
    # file download and extraction
    for link in links:
        url = "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/" + link
        tgzfile = urllib.request.urlretrieve(url, dirsave + os.sep + link)
        subprocess.call("tar -xvzf " + dirsave + os.sep + "*.tgz" + " -C " + dirsave, shell = True)
        subprocess.call("rm " + dirsave + os.sep + "*.tgz", shell = True)
# # notebook version:
#        !cd $DATA_RAW && tar -xvzf ./*.tgz
#        !cd $DATA_RAW && rm ./*.tgz


def get_gender(dirname):
    fname = os.path.join(dirname, 'etc', 'README')
    try:
        f = open(fname, 'r', encoding="utf-8")
        for line in f.readlines():
            if line.startswith("Gender"):
                return (line.strip().split(': ')[1].upper().strip('[];')[0])
    except:
        return 'U'


def df_init():
    fileids = os.listdir(DATA_RAW)
    genders = [get_gender(DATA_RAW + os.sep + dirname) for dirname in fileids]
    df = pd.DataFrame({'id': fileids, 'gender': genders})
    # ['MALE', 'FEMALE', 'WEIBLICH', None, 'UNKNOWN', 'MAKE','PLEASE SELECT', 'ADULT', 'U', 'MASCULINO']
    df.gender = df.gender.map({'W': 'F', 'P': 'U', 'A': 'U', 'P': 'U', 'M': 'M', 'F': 'F', 'U': 'U'})
    df.gender = np.where((~df.gender.isin(['M', 'F', 'U'])), 'U', df.gender)
    df[df['gender'] != 'U'].to_csv(PROCESSED + os.sep + 'df_init.csv', sep=';', index=False)
    # return df

def freq_amp(y, fs):

    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    # exclude irrelavent spectral range and 60 Hz (6) noise from electrical signals due to AC power-line contamination (i.e. too close to transformers).
    # 120 hz: bridge rectifier 47
    human_range = (freq >= 70) & (freq <= 300)

    # freq and amp with cutoffs
    amp = normalize_spec(spec[human_range])
    freq = freq[human_range]
    return freq, amp


def spectral_statistics(amp, freq):
    """
    Compute mean, median, std, q1, q3, inter-q, skewness, kurtosis, peak frequency in [Hz]
    :param y: 1-d signal
    :param fs: sampling frequency [Hz]
    :return: dictionary of mean, median, std, q1, q3, inter-q, skewness, kurtosis, peak.
    """

    def get_quantile(freq, amp, quantile):
        idxmin = np.argmin(np.abs(amp.cumsum() - quantile))
        freq_q = freq[idxmin]
        return freq_q
    try:
        median = get_quantile(freq, amp, 0.5)
        q1 = get_quantile(freq, amp, 0.25)
        q3 = get_quantile(freq, amp, 0.75)
        interq = q3 - q1
        peak = freq[np.argmax(amp)]
        mean = (freq * amp).sum()
        std = np.sqrt(((freq - mean) ** 2 * amp).sum())
        skewness = (((freq - mean) ** 3 * amp).sum()) / (std ** 3)
        kurtosis = (((freq - mean) ** 4 * amp).sum()) / (std ** 4)

        low = (freq < 180)
        high = (freq >= 180)
        pl = amp[low].max()
        xl = freq[low][amp[low].argmax()]
        ph = amp[high].max()
        xh = freq[high][amp[high].argmax()]
        h2l = ph / pl
        h2l_submin = (ph - amp.min()) / (pl - amp.min())

        return pd.DataFrame({'mean': [mean],
                             'median': [median],
                             'std': [std],
                             'q1': [q1],
                             'q3': [q3],
                             'inter_q': [interq],
                             'skewness': [skewness],
                             'kurtosis': [kurtosis],
                             'peak_lo': [xl],
                             'peak_hi': [xh],
                             'pratio': [h2l],
                             'pratio_submin': [h2l_submin]
                             })
    except:
        return pd.DataFrame({'mean': [np.nan],
                             'median': [np.nan],
                             'std': [np.nan],
                             'q1': [np.nan],
                             'q3': [np.nan],
                             'inter_q': [np.nan],
                             'skewness': [np.nan],
                             'kurtosis': [np.nan],
                             'peak_lo': [np.nan],
                             'peak_hi': [np.nan],
                             'pratio': [np.nan],
                             'pratio_submin': [np.nan]
                             })

def normalize_spec(spec):
    return spec / spec.sum()
def get_stats(dirname):
    try:
        flist = os.listdir(dirname)
        datas = [scipy.io.wavfile.read(dirname + os.sep + fname) for fname in flist]
        falist = [freq_amp(y, rate) for (rate, y) in datas]
        df_fa = pd.concat([pd.DataFrame({'freq': x, 'amp': y}) for x, y in falist])
        df_fa = df_fa.sort_values(by='freq').set_index('freq').rolling(500, center=True).mean().reset_index()
        df_fa.dropna(axis=0, inplace=True)
        df_fa.amp = normalize_spec(df_fa.amp)
        return spectral_statistics(df_fa.amp.values, df_fa.freq.values)
    except:
        return spectral_statistics([np.nan], [np.nan])

def get_dfstats(argin):
    df, DATA_RAW, PROCESSED, num = argin
    dfs = np.array_split(df, 10)
    num = 0
    for df in dfs:
        df_stats = [get_stats(DATA_RAW + os.sep + fname + os.sep + 'wav') for fname in df.id]
        df_stats = pd.concat(df_stats, ignore_index=True)
        df_stats.index = df.index
        df_out = pd.concat([df, df_stats], axis = 1)
        df_out.to_csv(PROCESSED + os.sep+'stats'+os.sep  + str(num)+'.csv', index = False)
        num +=1

    df_outs = [pd.read_csv(PROCESSED + os.sep + 'stats' + os.sep + str(num) + '.csv') for num in range(10)]
    df_out = pd.concat(df_outs, ignore_index= True)
    df_out.to_csv(PROCESSED + os.sep + 'stats_summary.csv', index = False)


def make_data_stats(multiproc = False):
    tic = time.time()
    df_init()
    df = pd.read_csv(PROCESSED + os.sep + 'df_init.csv', sep=';')
    if multiproc == True:
        import multiprocessing
        from multiprocessing import Pool
        #import workers
        multiprocessing.freeze_support()
        n_cores = multiprocessing.cpu_count()
        pool = Pool(n_cores)
        pool.map(get_dfstats, [(df,DATA_RAW,PROCESSED,0)])

    else:
        get_dfstats((df,DATA_RAW,PROCESSED,0))

    toc = time.time()
    print(toc-tic)

def get_freq_amplist(dirname):
    flist = os.listdir(dirname)
    datas = [scipy.io.wavfile.read(dirname+ os.sep + fname) for fname in flist]
    falist = [freq_amp(y, rate) for (rate, y) in datas]
    return falist
def plot_track(df, ax, i, j):
    tic = time.time()
    fname = df.id[j]
    g = df.gender[j]
    title = ('{}: {}'.format(fname, g))
    import scipy.signal as sig
    import statsmodels.api as sm

    dirname = DATA_RAW + os.sep + fname + os.sep + 'wav'
    falist = get_freq_amplist(dirname)
    df_fa = pd.concat([pd.DataFrame({'freq': x, 'amp': y}) for x, y in falist])
    df_fa = df_fa.sort_values(by='freq').set_index('freq').rolling(500, center=True).mean().reset_index()
    df_fa.dropna(axis=0, inplace=True)

    ax.plot(df_fa.freq, df_fa.amp)
    ax.plot(np.array([180, 180]), np.array([0.000, df_fa.amp.values.max()]), linestyle='-', color='black', lw=1)
    ax.set_title(title)
    ax.set_xlim([70, 300])


def fig_10samples():
    df = pd.read_csv(PROCESSED + os.sep + 'df_init.csv', sep=';')

    f, ax = plt.subplots(10, 1)
    _ = f.set_figheight(15)  # (figsize = (40,60))
    _ = f.set_figwidth(10)

    for i in range(10):
        plot_track(df, ax[i], i, i)
    plt.tight_layout()
    fname = PROCESSED + os.sep + 'plots' + os.sep + 'samples_first10.png'
    plt.savefig(fname, dpi=100)


if __name__ == "__main__":
    # print(os.listdir(PROCESSED))
    #subprocess.check_output(["cd",  DATA_RAW, "&&", "ls", "-l"])
    if sys.argv[-1] == "1":
        download_extract(DATA_RAW)
    make_data_stats()