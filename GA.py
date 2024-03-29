import copy
import numpy as np

from colorama import init
from termcolor import colored

# from TS import TSScheduler


class GAScheduler:
    def __init__(self, pt, jt):
        init()  # init colorama
        self.pt = np.array(pt)
        self.jt = np.array(jt)

        self.num_job = self.pt.shape[0]  # number of jobs
        print(colored("[scheduler]", "blue"), "number of jobs:", self.num_job)
        self.num_mc = 4  # number of machines
        self.num_gene = self.num_job   # number of genes in a chromosome

        self.crossover_rate = 1
        self.mutation_rate = 0.2

    # initialize population
    def init_population(self, population_size):
        population_list = []
        for i in range(population_size):
            # generate a random chromosome
            random_chromosome = [
                np.random.permutation(
                    np.arange(1, self.num_job + 1)
                ).tolist(),  # operation sequence
                np.random.randint(
                    1, self.num_mc + 1, size=self.num_job
                ).tolist(),  # machine selection
            ]
            population_list.append(random_chromosome)
        return population_list

    # one point crossover - candidate-order based
    def crossover(self, population_size, population_list):
        # sort array A according to B
        def relativeSortArray(A, B, cutpoint):
            new_A = copy.deepcopy(A)
            # sort OS according to the sequence in B
            new_A[0][cutpoint:] = sorted(
                new_A[0][cutpoint:], key=(B + sorted(new_A[0][cutpoint:])).index
            )
            # sort MS according to how OS is sorted
            indices = [A[0].index(element) for element in new_A[0]]
            temp = np.array(new_A[1])
            new_A[1] = temp[indices].tolist()
            # mutate (choose a candidate with largest reference point)
            mutation_prob_1 = np.random.rand()
            if self.mutation_rate >= mutation_prob_1:
                new_A[0][cutpoint:] = np.roll(new_A[0][cutpoint:], 1)
                new_A[1][cutpoint:] = np.roll(new_A[1][cutpoint:], 1)
            # mutate (random machine selection)
            mutation_prob_2 = np.random.rand()
            if self.mutation_rate >= mutation_prob_2:
                op_to_be_changed = np.random.randint(0, len(new_A[1]))
                new_A[1][op_to_be_changed] = np.random.randint(1, self.num_mc + 1)
            return new_A

        parent_list = copy.deepcopy(population_list)
        offspring_list = copy.deepcopy(population_list)
        # generate a random sequence to select the parent chromosomes to crossover
        S = list(np.random.permutation(population_size))

        for m in range(int(population_size / 2)):
            crossover_prob = np.random.rand()
            if self.crossover_rate >= crossover_prob:
                parent_1 = population_list[S[2 * m]]
                parent_2 = population_list[S[2 * m + 1]]
                child_1 = parent_1
                child_2 = parent_2
                cutpoint = np.random.randint(0, self.num_gene)

                child_1 = relativeSortArray(child_1, child_2[0], cutpoint)
                child_2 = relativeSortArray(child_2, child_1[0], cutpoint)

                offspring_list[S[2 * m]] = child_1
                offspring_list[S[2 * m + 1]] = child_2

        return parent_list, offspring_list

    # fitness calculation (makespan)
    def fitness(self, population_size, parent_list, offspring_list):
        total_chromosome = copy.deepcopy(parent_list) + copy.deepcopy(
            offspring_list
        )  # parent and offspring chromosomes combination
        chrom_fitness, chrom_fit = [], []
        total_fitness = 0

        for m in range(population_size * 2):
            time = [
                0 for _ in range(self.num_mc)
            ]  # record the total processing time of each machine
            job_type = [
                [] for _ in range(self.num_mc)
            ]  # record the sequence of job type of each machine

            for job, machine in zip(total_chromosome[m][0], total_chromosome[m][1]):
                processing_time = int(self.pt[job - 1][0])
                type = int(self.jt[job - 1][0])
                setup_time = int(self.jt[job - 1][1])

                if np.size(job_type[machine - 1]) == 0:  # first job on this machine
                    time[machine - 1] += setup_time + processing_time
                    job_type[machine - 1].append(type)
                else:
                    # get the last job type on this machine to determine if setup is needed
                    last_type = job_type[machine - 1][-1]
                    if type == last_type:
                        time[machine - 1] += processing_time
                    else:
                        time[machine - 1] += setup_time + processing_time
                    job_type[machine - 1].append(type)

            makespan = np.max(time)
            chrom_fitness.append(1 / makespan)  # reciprocal of makespan
            chrom_fit.append(makespan)
            total_fitness = total_fitness + chrom_fitness[m]
        return total_chromosome, chrom_fitness, total_fitness, chrom_fit

    # Roulette wheel selection
    def select(
        self,
        population_size,
        population_list,
        total_chromosome,
        chrom_fitness,
        total_fitness,
    ):
        pk, qk = [], []

        for i in range(population_size * 2):
            pk.append(chrom_fitness[i] / total_fitness)
        for i in range(population_size * 2):
            cumulative = 0
            for j in range(0, i + 1):
                cumulative = cumulative + pk[j]
            qk.append(cumulative)

        selection_rand = [np.random.rand() for i in range(population_size)]

        for i in range(population_size):
            if selection_rand[i] <= qk[0]:
                population_list[i] = copy.deepcopy(total_chromosome[0])
            else:
                for j in range(0, population_size * 2 - 1):
                    if selection_rand[i] > qk[j] and selection_rand[i] <= qk[j + 1]:
                        population_list[i] = copy.deepcopy(total_chromosome[j + 1])
                        break
        return population_list

    # called in main to run GA
    def run_genetic(
        self,
        population_size=100,
        num_iteration=100,
        verbose=False,
        num_job=20,
        num_mc=3,
    ):
        assert population_size > 0, num_iteration > 0

        # initialize population (random generation)
        print(colored("[scheduler]", "blue"), "initializing population...")
        population_list = self.init_population(population_size)
        print(colored("[scheduler]", "blue"), "finished initializing population.")
        makespan_record = []
        Tbest = 999999999999999

        for current_generation in range(num_iteration):
            Tbest_now = 999999999999999
            # crossover
            print(
                colored("[evolving]", "green"),
                "evolving",
                current_generation + 1,
                "generation",
            )
            parent_list, offspring_list = self.crossover(
                population_size, population_list
            )
            # # apply tabu search to improve the quality of population
            # for i in range(population_size):
            #     t = TSScheduler(offspring_list[i], self.num_mc, self.pt, self.jt)
            #     new_t = t.run_TS()
            #     offspring_list[i] = copy.deepcopy(new_t)
            #     del t
            # fitness calculation

            total_chromosome, chrom_fitness, total_fitness, chrom_fit = self.fitness(
                population_size, parent_list, offspring_list
            )
            # selection
            population_list = self.select(
                population_size,
                population_list,
                total_chromosome,
                chrom_fitness,
                total_fitness,
            )
            # record the best solution
            for i in range(population_size * 2):
                if chrom_fit[i] < Tbest_now:
                    Tbest_now = chrom_fit[i]
                    sequence_now = copy.deepcopy(total_chromosome[i])
            if Tbest_now < Tbest:
                Tbest = Tbest_now
                sequence_best = copy.deepcopy(sequence_now)
                print(
                    colored("[better solution]", "cyan"),
                    "better solution found, current best time is",
                    Tbest_now,
                )

            makespan_record.append(Tbest)

        import matplotlib.pyplot as plt

        plt.plot([i for i in range(len(makespan_record))], makespan_record, "b")
        plt.ylabel("makespan", fontsize=15)
        plt.xlabel("generation", fontsize=15)
        plt.savefig("makespan-generation")

        return sequence_best, Tbest

    def draw_Gnatt(self, sequence_best):
        import plotly.figure_factory as ff
        from plotly.offline import plot
        import datetime

        time = [0 for _ in range(self.num_mc)]
        job_type = [[] for _ in range(self.num_mc)]
        # job_record and setup_record are dictionary, with (job, machine) as key and [start_time, end_time] as value
        job_record = {}
        setup_record = {}

        for job, machine in zip(sequence_best[0], sequence_best[1]):
            processing_time = int(self.pt[job - 1][0])
            type = int(self.jt[job - 1][0])
            setup_time = int(self.jt[job - 1][1])

            start_sec = time[machine - 1]
            setup_completed_time = time[machine - 1]
            if np.size(job_type[machine - 1]) == 0:
                setup_completed_time += setup_time
                time[machine - 1] += setup_time + processing_time
                job_type[machine - 1].append(type)
            else:
                last_type = job_type[machine - 1][-1]
                if type == last_type:
                    time[machine - 1] += processing_time
                else:
                    setup_completed_time += setup_time
                    time[machine - 1] += setup_time + processing_time
                job_type[machine - 1].append(type)

            if start_sec == setup_completed_time:
                start_time = str(datetime.timedelta(seconds=start_sec))
                end_time = str(datetime.timedelta(seconds=time[machine - 1]))
                job_record[(job, machine)] = [start_time, end_time]
            else:
                start_time = str(datetime.timedelta(seconds=start_sec))
                end_time = str(datetime.timedelta(seconds=setup_completed_time))
                setup_record[(job, machine)] = [start_time, end_time]
                start_time = str(datetime.timedelta(seconds=setup_completed_time))
                end_time = str(datetime.timedelta(seconds=time[machine - 1]))
                job_record[(job, machine)] = [start_time, end_time]

        df = []
        for machine in range(1, self.num_mc + 1):
            for job in range(1, self.num_job + 1):
                if (job, machine) in setup_record:
                    df.append(
                        dict(
                            Task="Machine %s" % (machine),
                            Start="2023-07-01 %s"
                            % (str(setup_record[(job, machine)][0])),
                            Finish="2023-07-01 %s"
                            % (str(setup_record[(job, machine)][1])),
                            Resource="Setup",
                        )
                    )
                if (job, machine) in job_record:
                    df.append(
                        dict(
                            Task="Machine %s" % (machine),
                            Start="2023-07-01 %s"
                            % (str(job_record[(job, machine)][0])),
                            Finish="2023-07-01 %s"
                            % (str(job_record[(job, machine)][1])),
                            Resource="Job %s" % (job),
                        )
                    )

        # create additional colors since default colors of Plotly are limited to 10 different colors
        r = lambda: np.random.randint(0, 255)
        colors = ["#%02X%02X%02X" % (r(), r(), r())]
        for _ in range(1, len(df) + 1):
            colors.append("#%02X%02X%02X" % (r(), r(), r()))
        fig = ff.create_gantt(
            df,
            colors=colors,
            index_col="Resource",
            show_colorbar=True,
            group_tasks=True,
            showgrid_x=True,
            title="Job shop Schedule",
        )
        plot(fig, filename="GA_job_shop_scheduling.html")
