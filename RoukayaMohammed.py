from mido import Message
from mido import MidiFile
from mido import MidiTrack
from music21.converter import parse
import random
import numpy as np

# Global variables
beat = 384
octave = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
major = [2, 4, 5, 7, 9, 11]
minor = [2, 3, 5, 7, 8, 10]

def note_to_index(note):
    """
    This function is responsible for getting the index of a given note in an octave

    :param note: char: a note char
    :return: integer: index of the note in the octave
    """
    for i in range(len(octave)):
        if note == octave[i]:
            return i

#These are different types of chords that can be generated for the accompaniment using a root key

def major_triad(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root, root + 4, root + 7]


def minor_triad(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root, root + 3, root + 7]


def first_inverted_major(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root + 12, root + 4, root + 7]


def first_inverted_minor(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root + 12, root + 3, root + 7]


def second_inverted_major(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root + 12, root + 16, root + 7]


def second_inverted_minor(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root + 12, root + 15, root + 7]


def diminished(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root, root + 3, root + 6]


def sus2(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root, root + 2, root + 7]


def sus4(root):
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [root, root + 5, root + 7]


def rest():
    """
    :param root: integer: a note value
    :return: list[int]: given chord for root
    """
    return [-1, -1, -1]


def avg_note(notes_in_beat):
    """
    This function is responsible for calculating the average note played in a quarter-bar

    :param notes_in_beat: list[int,int]: that contains data of notes and time in quarter-bar
    :return: float: that is the average note in quarter-bar
    """
    total_note = 0.0
    total_time = 0
    for note in notes_in_beat:
        total_time = total_time + note[1]
        total_note = total_note + (note[0] * (note[1]))
    return total_note / total_time


def average_notes(original_melody):
    """
    This function is responsible for calculating the mean of the notes in each quarter-bar in the melody

    :param original_melody: list[list[int}]: that contains data of notes and time in the original melody
    :return: list[float]: list includes the average notes for each quarter-bar in the input
    """
    avgs = []
    notes_in_beat = []
    time_in_beat = 0

    for tone in original_melody:
        time_in_beat = time_in_beat + tone[1]
        if tone[0] == 0:
            continue
        if time_in_beat == beat:
            notes_in_beat.append(tone)
            avgs.append(avg_note(notes_in_beat))
            notes_in_beat = []
            time_in_beat = 0
        if time_in_beat < beat:
            notes_in_beat.append(tone)
            continue
        if time_in_beat >= beat:
            notes_in_beat.append([tone[0], tone[1] - (time_in_beat - beat)])
            avgs.append(avg_note(notes_in_beat))
            time_in_beat = tone[1] - (notes_in_beat[len(notes_in_beat) - 1][1])
            while time_in_beat >= beat:
                avgs.append(float(tone[0]))
                time_in_beat = time_in_beat - beat
            notes_in_beat = []
            if time_in_beat != 0:
                notes_in_beat.append([tone[0], time_in_beat])
    return avgs


def spawn_individuals(size, length):
    """
    This function is responsible for spawning the individuals of the first generation

    :param size: integer: the size of the population
    :param length: integer: the number of chords in an individual
    :return: list[list[int]]: population of the first generation
    """
    population = []
    for i in range(size):
        individual = []
        for j in range(length):
            root = random.randint(0, 110)
            choose = random.randint(0, 10)
            match choose:
                case 0:
                    individual.append(major_triad(root))
                case 1:
                    individual.append(minor_triad(root))
                case 2:
                    individual.append(first_inverted_major(root))
                case 3:
                    individual.append(first_inverted_minor(root))
                case 4:
                    individual.append(second_inverted_major(root))
                case 5:
                    individual.append(second_inverted_minor(root))
                case 6:
                    individual.append(sus2(root))
                case 7:
                    individual.append(sus4(root))
                case 8:
                    individual.append(diminished(root))
                case _:
                    individual.append(rest())
        population.append(individual)
    return population


def similar(individual, avg):
    """
    This function is calculates he similarity of each chord with the average note played on the 
    corresponding quarter of a bar. The weights are obtained by experimentation.

    :param individual: list[int]: the individual for which the similarity value is to be calculated
    :param avg: list[float]: the average notes played in each quarter of a bar in the original melody
    :return: float: the similarity value of the individual
    """
    value = 0
    for chord in range(len(individual)):
        diff1 = max(10.0 - abs(float(avg[chord]) - float(individual[chord][0])), 0.0)
        diff2 = max(10.0 - abs(float(avg[chord]) - float(individual[chord][1])), 0.0)
        diff3 = max(10.0 - abs(float(avg[chord]) - float(individual[chord][2])), 0.0)
        value = value + diff1 * 5 + diff2 * 10.0 + diff3 * 5
    return value


def in_scale(individual, scale):
    """
    This function is responsible for calculating the value of existence of the chords of a given 
    individual in the scale of the original melody

    :param individual: list[int]: the individual for which the in_scale value is to be calculated
    :param scale: list[char]: the scale of the original_melody
    :return: float: the in_scale value for the individual
    """
    value = 0
    for chord in individual:
        tonic = chord[0] % 12
        subdominant = chord[1] % 12
        dominant = chord[2] % 12
        tonic_check = False
        subdominant_check = False
        dominant_check = False
        for j in range(len(scale)):
            note = note_to_index(scale[j])
            if tonic == note:
                tonic_check = True
            if subdominant == note:
                subdominant_check = True
            if dominant == note:
                dominant_check = True
        matches = 0
        if subdominant_check:
            matches = matches + 1
        if tonic_check:
             matches = matches + 1
        if dominant_check:
            matches = matches + 1
        if matches >= 2:
            value += 1
        else:
            value -= 1
    return value * 100


def fitness(individual, avg, scale):
    """
    This function is used to get the fitness value of an individual.
    The individual’s fitness value depends on two things:
        1- The similarity of each chord with the average note played on the corresponding quarter of a bar.
        2- The existence of this chord in the scale of the detected key of the melody.

    :param individual: list[int]: the individual for which the fittness value is to be calculated
    :param avg: list[float]: includes the average notes played for each quarter-bar in the melody
    :param scale: list[char]: the scale of the original_melody
    :return: float: the fitness value of the individual
    """
    return in_scale(individual, scale) + similar(individual, avg)


def selection(population, avg, scale):
    """
    This function is responsible for the selection process where it calculates the fitness value for the 
    whole population, then sorts them and the fittest half survives for the next generation.

    :param population: list[list[int]]: the current population to go into selection
    :param avg: list[float]: the average notes played in each quarter of a bar in the original melody
    :param scale: list[char]: the scale of the original_melody
    :return: list[list[int]]: the fittest individuals for the next generation
    """
    population = np.random.permutation(population)
    contestants = []
    for contestantID in range(len(population)):
        contestants.append([fitness(population[contestantID], avg, scale), contestantID])
    contestants = sorted(contestants, reverse=True)
    selected = []
    for i in range(len(contestants)//2):
        selected.append(population[contestants[i][1]])
    return selected


def crossover(population):
    """
    This function is responsible for the crossover technique used is N point crossover 
    where N is the length of a single individual.
    Each child is the exact opposite of the other.

    :param population: list[list[int]]: the current population to go into mating phase
    :return: list[list[int]]: the new population after the mating phase
    """
    first_generation = np.random.permutation(population)
    next_generation = []
    for individual in range(0, len(first_generation), 2):
        next_generation.append(first_generation[individual])
        if individual + 1 == len(first_generation):
            break
        child1 = []
        child2 = []
        next_generation.append(first_generation[individual + 1])
        for i in range(len(first_generation[individual])):
            child1.append(first_generation[individual + (i % 2)][i])
            child2.append(first_generation[individual + ((i + 1) %2)][i])
        probability_of_mutation = random.uniform(0, 1)
        if (probability_of_mutation < 0.15):
            next_generation.append(mutation(child1))
            next_generation.append(mutation(child2))
        else:
            next_generation.append(child1)
            next_generation.append(child2)
    return next_generation


def mutation(individual):
    """
    This function is used to combine both inversion with the bit-flip mutation techniques.

    :param individual: list[int]: individual that was selected for mutation
    :return: list[int]: the same individual after mutation
    """
    individual = np.random.permutation(individual)
    chord = random.randint(0, len(individual) - 1)
    root = individual[chord][0]
    choose = random.randint(0, 10)
    match choose:
        case 0:
            individual[chord] = major_triad(root)
        case 1:
            individual[chord] = minor_triad(root)
        case 2:
            individual[chord] = first_inverted_major(root)
        case 3:
            individual[chord] = first_inverted_minor(root)
        case 4:
            individual[chord] = second_inverted_major(root)
        case 5:
            individual[chord] = second_inverted_minor(root)
        case 6:
            individual[chord] = sus2(root)
        case 7:
            individual[chord] = sus4(root)
        case 8:
            individual[chord] = diminished(root)
        case _:
            individual[chord] = rest()

    return individual


def get_best_individual(population, average, scale):
    """
    This function is responsible for analyzing the population after the evolution process and get the fittest_individual 
    among the generations

    :param population: list[list[int]]: the population of the last generation
    :param average: list[float]: includes the average notes played for each quarter-bar in the melody
    :param scale: list[char]: the scale of the original_melody
    :return: list[int]: the fittest individual
    """
    fittest_individual = []
    max_fitness = 0
    for individual in population:
        individual_score = fitness(individual, average, scale)
        if individual_score > max_fitness:
            fittest_individual = individual
            max_fitness = individual_score
    return fittest_individual


def major_key_notes(key):
    """
    This function is used to produce the major scale pattern for the given root key
    :param key: Key object: root key to produce the major scale
    :return: list[char]: the list consisting of the notes of the major scale pattern
    """
    scale = [key]
    key_index = note_to_index(key)
    for i in range(6):
        scale.append(octave[(key_index + major[i]) % 12])
    return [key, scale]


def minor_key_notes(key):
    """
    This function is used to produce the minor scale pattern for the given root key
    :param key: Key object: root key to produce the minor scale
    :return: list[char]: the list consisting of the notes of the minor scale pattern
    """
    scale = [key]
    key_index = note_to_index(key)
    for i in range(6):
        scale.append(octave[(key_index + minor[i]) % 12])
    return [key, scale]


def get_scale(input):
    """
    This function is used to parse the input and identify the key for the given melody and get it's scale to be used in 
    composing chords for the melody.

    :param input: string: the input midi file name
    :return: list[char]: the list consisting of the notes of the scale of the key detected
    """

    scale = []
    major_keys = []
    for i in range(len(octave)):
        major_keys.append(major_key_notes(octave[i]))

    minor_keys = []
    for i in range(len(octave)):
        minor.append(minor_key_notes(octave[i]))

    input_song = parse(input)
    key = input_song.analyze('key')
    print(key)
    root_note = str(key).split()[0].capitalize()
    
    if key.type == 'major':
        for i in range(len(major_keys)):
            if major_keys[i][0] == root_note:
                scale = major_keys[i][1]
    else:
        for i in range(len(minor_keys)):
            if minor_keys[i][0] == root_note:
                scale = minor_keys[i][1]
    return scale


def write_output(input, best_individual, output, key):
    """
    This is the output function.
    It's responsible for forming the accompaniment track using the best_individual 
    solution we got among the generations and produce a midi file of the result.

    :param input: mido.Midifile: that contains the input midi file
    :param best_individual: list[int,int,int]: the fittest individual among the generations
    :param output: string: output file name
    :param key: Key object: that has the identified key for the input melody
    :return:
    """
    output += (key + ".mid")
    generated_output = MidiFile()
    generated_track = MidiTrack()
    for i in range(2):
        generated_output.tracks.append(input.tracks[i])

    generated_track.append(input.tracks[1][0])
    generated_output.ticks_per_beat = input.ticks_per_beat
    rest_time = 0
    for chord in best_individual:
        if chord[0] == 0:
            rest_time = rest_time + beat
            continue
        generated_track.append(
            Message("note_on", channel=0, note=chord[0], velocity=50, time=rest_time))
        generated_track.append(
            Message("note_on", channel=0, note=chord[1], velocity=50, time=0))
        generated_track.append(
            Message("note_on", channel=0, note=chord[2], velocity=50, time=0))
        generated_track.append(
            Message("note_off", channel=0, note=chord[0], velocity=0, time=beat))
        generated_track.append(
            Message("note_off", channel=0, note=chord[1], velocity=0, time=0))
        generated_track.append(
            Message("note_off", channel=0, note=chord[2], velocity=0, time=0))
        rest_time = 0

    generated_track.append(input.tracks[1][-1])
    generated_output.tracks.append(generated_track)
    generated_output.save(output)
    return


def main(input, output):
    """
    main function responsible for most of initializtion and the evolution process
    :param input: string: the input midi file name
    :param output: string: the output midi file name
    :return:
    """
    input_mid = MidiFile(input, clip=True)
    input_song = parse(input)
    key = input_song.analyze('key')
    key = str(key)

    number_of_generations = 600

    size_of_population = 1024

    original_melody = []
    for track in input_mid.tracks:
        for message in track:
            if message.time != 0:
                if message.type == "note_on":
                    original_melody.append([0, message.time])
                else:
                    original_melody.append([message.note, message.time])

    scale = get_scale(input)
    average = average_notes(original_melody)
    population = spawn_individuals(size_of_population, len(average))

    for i in range(number_of_generations):
        population = crossover(selection(population, average, scale))

    best_individual = get_best_individual(population, average, scale)

    write_output(input_mid, best_individual, output, key)

    return


if __name__ == '__main__':
    main("Input/input1.mid", "RoukayaMohammedOutput1-")
    main("Input/input2.mid", "RoukayaMohammedOutput2-")
    main("Input/input3.mid", "RoukayaMohammedOutput3-")
