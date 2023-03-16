import mido
import string
import np
mid = mido.MidiFile('barbiegirl_mono.mid', clip=True)
mid.tracks
def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))
    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]
def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 109 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note+12] = velocity if on_ else 0
    return result

def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]


def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]
def track2seq(track):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result
result_array = mid2arry(mid)
import matplotlib.pyplot as plt
plt.plot(range(result_array.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.title("barbiegirl_mono.mid")
plt.show()
answer=[]
time=[]
i=0
n=len(result_array)
m=len(result_array[0])
while i < n-2:
    j=0
    is1=0
    while j<m:
        if result_array[i][j]!=0:
         is1=1
         break
        j=j+1
    if is1==0:
        if answer[len(answer)-1]==-1:
            time[len(time)-1]=time[len(time)-1]+1
        else:
            time.append(1)
            answer.append(-1)
        i=i+1
        continue
    answer.append(j)
    count=0
    while i<n-1 and result_array[i+1][j]!=0:
        i=i+1
        count=count+1
    time.append(count)
    i=i+1

time[len(time)-1]=time[len(time)-1]+1
print (answer)
print (time)
import music21
score = music21.converter.parse('barbiegirl_mono.mid')
key = score.analyze('key')
print(key.tonic, key.mode)
time_duration=0
for i in time:
    time_duration=time_duration+i
count1=0
answer1=[]
answer1.append(answer[0])
j=-1
for i in time:
    j=j+1
    if count1>370 and j!=0:
        if answer[j]!=-1:
            answer1.append(answer[j])
        else:
            answer1.append(answer1[len(answer1)-1])
        count1=0
    count1 = count1 + i
print(answer1)

"""
def arry2mid(ary, tempo=500000):
    # get the difference
    new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
    changes = new_ary[1:] - new_ary[:-1]
    # create a midi file with an empty track
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    # add difference in the empty track
    last_time = 0
    for ch in changes:
        if set(ch) == {0}:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_vol = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_vol):
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
                first_ = False
            last_time = 0
    return mid_new

mid_new = arry2mid(result_array, 545455)
mid_new.save('mid_new.mid')
"""

import random
import math
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from lily_template import TEMPLATE
"""
si la armadura tiene # se multiplica 7 por el número de sostenidos mod(12)
si la armadura tiene b se multiplica 5 por el número de bemoles mod(12)
"""

# Global Variables
OPTIONS_M = ((0, -3, 5),
             (0, -3, 5),
             (0, -4, 5),
             (0, -3, 6),
             (0, -3, 5),
             (0, -4, 5),
             (0, -4, 5)
             )
OPTIONS_m = ((0, -4, 5),
             (0, -4, 5),
             (0, -3, 5),
             (0, -3, 5),
             (0, -4, 5),
             (0, -3, 6),
             (0, 5)
             )
MOD_M = ('M', 'm', 'm', 'M', 'M', 'm', 'd')
MOD_m = ('m', 'd', 'M', 'm', 'M', 'M', 'M')


def neighborhood(iterable):
    """Generator gives the prev actual and next"""
    iterator = iter(iterable)
    prev = None
    item = next(iterator)  # throws StopIteration if empty.
    for nex in iterator:
        yield (prev, item, nex)
        prev = item
        item = nex
    yield (prev, item, None)


def setTon(line):
    """Determine the tonality of the exercice"""
    ton = line[:2]
    notes = list(map(int, line[3:].split(' ')))
    if ton[1] == '#':
        ton = (int(ton[0]) * 7) % 12
    else:
        ton = (int(ton[0]) * 5) % 12
    for note in notes:
        if (ton + 6) % 12 == note % 12:
            ton = str((ton - 3) % 12) + 'm'
            break
    else:
        if ton - 3 == notes[-1] % 12:
            ton = str((ton - 3) % 12) + 'm'
        else:
            ton = str(ton) + 'M'
    return ton, notes


def creatChord(nameC, noteF):
    """Create one chord given the name of the chord and the fundamental note"""
    num_funda = int(nameC[:-1])
    if nameC[-1] == 'M':
        val_notes = [num_funda, (num_funda + 4) % 12, (num_funda + 7) % 12]
    elif nameC[-1] == 'm':
        val_notes = [num_funda, (num_funda + 3) % 12, (num_funda + 7) % 12]
    elif nameC[-1] == 'd':
        val_notes = [num_funda, (num_funda + 3) % 12, (num_funda + 6) % 12]

    tenorR = list(range(48, 69))
    contR = list(range(52, 77))
    sopR = list(range(60, 86))

    # Depending in the bass note this are the options for the others voices
    if noteF % 12 == val_notes[0]:
        opc = [[1, 1, 1], [2, 1, 0], [0, 1, 2]]
    elif noteF % 12 == val_notes[1]:
        opc = [[1, 0, 2], [3, 0, 0], [2, 0, 1]]
    elif noteF % 12 == val_notes[2]:
        opc = [[1, 1, 1], [2, 1, 0]]

    opc = random.choice(opc)
    chordN = list()
    for num, val in zip(opc, val_notes):
        chordN += [val] * num

    random.shuffle(chordN)

    chord = [noteF, ]
    for nte, voce in zip(chordN, [tenorR, contR, sopR]):
        posible_n = [x for x in voce if x % 12 == nte]
        chord.append(random.choice(posible_n))

    return chord


def selChord(ton, notesBass):
    """Select the chords from all the posibilities"""
    listaOp = OPTIONS_M if ton[-1] == 'M' else OPTIONS_m
    listaMod = MOD_M if ton[-1] == 'M' else MOD_m
    prog = list()

    for note in notesBass:
        name = note % 12
        grad = name - int(ton[:-1])
        grad = math.ceil(((grad + 12) % 12) / 2)
        num = (listaOp[grad][random.randint(0, len(listaOp[grad]) - 1)]
               + name + 12) % 12
        grad = num - int(ton[:-1])
        grad = math.ceil(((grad + 12) % 12) / 2)
        name = '{}{}'.format(num, listaMod[grad])
        prog.append([creatChord(name, note), grad])
    return prog


def newChordProg(ton, notes):
    """Create a new individual given the tonality and the base notes"""
    chords = selChord(ton, notes)
    for c in chords:
        yield c


def check_interval(chord):
    """Return the number of mistakes in the distance between the notes"""
    res = 0
    if chord[2] - chord[1] > 12 or chord[2] - chord[1] < 0:
        res += 15
    if chord[3] - chord[2] > 12 or chord[3] - chord[2] < 0:
        res += 15

    if chord[1] == chord[2] or chord[2] == chord[3]:
        res += 1.4
    return res


def check_2_chords(ch1, ch2):
    """Return the number of mistakes in the intervals between 2 chords"""
    res = 0

    # Check for 5° and 8°
    ite1 = map(lambda x, y: y - x, ch1[:-1], ch1[1:])
    ite2 = map(lambda x, y: y - x, ch2[:-1], ch2[1:])
    for inter1, inter2 in zip(ite1, ite2):
        if inter1 == 7 and inter2 == 7:
            res += 15
        elif inter1 == 0 and inter2 == 0:
            res += 15
        elif inter1 == 12 and inter2 == 12:
            res += 15

    # Check for big intervals, just to make it more "human"
    for note1, note2 in zip(ch1[1:], ch2[1:]):
        if abs(note1 - note2) >= 7:  # 7 equals 5° interval
            res += .7

    return res


def evalNumErr(ton, individual):
    """Evaluation function"""
    res = 0
    for prev, item, nex in neighborhood(individual):
        res += check_interval(item[0])
        if prev is None:
            if item[1] != 0:
                res += 6
            continue
        else:
            if prev[1] in [4, 6] and item[1] in [3, 1]:
                res += 20
            res += check_2_chords(prev[0], item[0])
        if nex is None:
            if item[1] in [1, 2, 3, 4, 5, 6]:
                res += 6
    return (res,)


def mutChangeNotes(ton, individual, indpb):
    """Mutant function"""
    new_ind = toolbox.clone(individual)
    for x in range(len(individual[0])):
        if random.random() < indpb:

            listaOp = OPTIONS_M if ton[-1] == 'M' else OPTIONS_m
            listaMod = MOD_M if ton[-1] == 'M' else MOD_m

            note = individual[x][0][0]

            name = note % 12
            grad = name - int(ton[:-1])
            grad = math.ceil(((grad + 12) % 12) / 2)
            num = (listaOp[grad][random.randint(0, len(listaOp[grad]) - 1)]
                   + name + 12) % 12
            grad = num - int(ton[:-1])
            grad = math.ceil(((grad + 12) % 12) / 2)
            name = '{}{}'.format(num, listaMod[grad])

            new_ind[x] = [creatChord(name, note), grad]

    del new_ind.fitness.values
    return new_ind,


def transform_lilypond(ton, indiv):
    """Take one list of chords and print the it in lilypond notation"""
    note_map = dict()
    if ton[-1] == 'M':
        note_map = {0: 'c',
                    1: 'cis',
                    2: 'd',
                    3: 'dis',
                    4: 'e',
                    5: 'f',
                    6: 'fis',
                    7: 'g',
                    8: 'gis',
                    9: 'a',
                    10: 'ais',
                    11: 'b'
                    }
    else:
        note_map = {0: 'c',
                    1: 'des',
                    2: 'd',
                    3: 'ees',
                    4: 'e',
                    5: 'f',
                    6: 'ges',
                    7: 'g',
                    8: 'aes',
                    9: 'a',
                    10: 'bes',
                    11: 'b'
                    }
    voces = [[], [], [], []]

    for chord in indiv:
        for note, voce in zip(chord, voces):

            octave = (note // 12) - 4
            name_lily = note_map[note % 12]
            if octave < 0:
                name_lily += ',' * (octave * -1)
            elif octave > 0:
                name_lily += "'" * octave
            voce.append(name_lily)
    form_txt = '{}|\n{}|\n{}|\n{}|\n'
    print(form_txt.format(*(' '.join(voce) for voce in reversed(voces))))


def main(ton):
    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', numpy.mean)
    stats.register('std', numpy.std)
    stats.register('min', numpy.min)
    stats.register('max', numpy.max)

    pop, log = algorithms.eaSimple(pop,
                                   toolbox,
                                   cxpb=0.5,
                                   mutpb=0.3,
                                   ngen=70,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)
    while min(log.select('min')) > 15:
        pop = toolbox.population(n=400)
        pop, log = algorithms.eaSimple(pop,
                                       toolbox,
                                       cxpb=0.5,
                                       mutpb=0.3,
                                       ngen=70,
                                       stats=stats,
                                       halloffame=hof,
                                       verbose=True)

    for best in hof:
        print([x[0] for x in best],end='\n=======\n')

        transform_lilypond(ton, [x[0] for x in best])


if __name__ == '__main__':
    line = input('n[#b] notas ')
    ton, notes = setTon(line)
    print(ton, notes)

    # ========================= GA setup =========================
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register('creat_notes', newChordProg, ton, notes)
    toolbox.register('individual', tools.initIterate, creator.Individual,
                     toolbox.creat_notes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', evalNumErr, ton)
    toolbox.register('mate', tools.cxOnePoint)
    toolbox.register('mutate', mutChangeNotes, ton, indpb=0.4)
    toolbox.register('select', tools.selTournament, tournsize=3)
    # =============================================================

    main(ton)
