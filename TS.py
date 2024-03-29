import copy
import numpy as np

class TSScheduler:
    def __init__(self, chromosome, num_mc, pt, jt):
        self.num_mc = num_mc
        self.pt = np.array(pt)
        self.jt = np.array(jt)
        self.initial_chromosome = chromosome
        self.current_chromosome = chromosome
        self.tbest = 999999999999999
        self.neighborhood = []
        self.tabu_list = []

    def find_neighbors(self):
        length = len(self.current_chromosome[1])
        self.neighborhood = []
        for i in range(length):
            for j in range(i + 1, length):
                neighbor = copy.deepcopy(self.current_chromosome)
                neighbor[1][i], neighbor[1][j] = neighbor[1][j], neighbor[1][i]
                self.neighborhood.append(neighbor)

    def fitness(self):
        chrom_fit = []
        for m in range(len(self.neighborhood)):
            time = [0 for _ in range(self.num_mc)] # record the total processing time of each machine
            job_type = [[] for _ in range(self.num_mc)] # record the sequence of job type of each machine
            
            for job, machine in zip(self.neighborhood[m][0], self.neighborhood[m][1]):
                processing_time = int(self.pt[job-1][0])
                type            = int(self.jt[job-1][0])
                setup_time      = int(self.jt[job-1][1])

                if np.size(job_type[machine-1]) == 0 : # first job on this machine
                    time[machine-1] += setup_time + processing_time
                    job_type[machine-1].append(type)
                else :
                    # get the last job type on this machine to determine if setup is needed
                    last_type = job_type[machine-1][-1]
                    if type == last_type:
                        time[machine-1] += processing_time
                    else:
                        time[machine-1] += setup_time + processing_time
                    job_type[machine-1].append(type)
        
            makespan = np.max(time)
            chrom_fit.append(makespan)
        return chrom_fit
    
    def update(self, chrom_fit):
        def check_tabulist(chromosome):
            for i in range(len(self.tabu_list)):
                if np.array_equal(self.tabu_list[i], chromosome):
                    return False
            return True
        
        while(True):
            index_best = 0
            makespan_best = 999999999999999
            for i in range(len(chrom_fit)):
                if chrom_fit[i] < makespan_best:
                    makespan_best = chrom_fit[i]
                    index_best = i
            
            if check_tabulist(self.neighborhood[index_best]):
                self.current_chromosome = self.neighborhood[index_best]
                self.tabu_list.append(self.neighborhood[index_best])
                self.tbest = makespan_best
                break
            else:
                chrom_fit[index_best] = 999999999999999

    def run_TS(self):
        num_iteration = 7
        for i in range(num_iteration):
            self.find_neighbors()
            chrom_fit = self.fitness()
            self.update(chrom_fit)

        time = [0 for _ in range(self.num_mc)] # record the total processing time of each machine
        job_type = [[] for _ in range(self.num_mc)] # record the sequence of job type of each machine
        for job, machine in zip(self.initial_chromosome[0], self.initial_chromosome[1]):
            processing_time = int(self.pt[job-1][0])
            type            = int(self.jt[job-1][0])
            setup_time      = int(self.jt[job-1][1])
            if np.size(job_type[machine-1]) == 0 : # first job on this machine
                time[machine-1] += setup_time + processing_time
                job_type[machine-1].append(type)
            else :
                # get the last job type on this machine to determine if setup is needed
                last_type = job_type[machine-1][-1]
                if type == last_type:
                    time[machine-1] += processing_time
                else:
                    time[machine-1] += setup_time + processing_time
                job_type[machine-1].append(type)
        initial_makespan = np.max(time)

        if self.tbest < initial_makespan :
            return self.current_chromosome
        else :
            return self.initial_chromosome