from pathlib import Path


def parse(instance: str) -> dict:
    lines = instance.split("\n")

    firstLine = lines.pop(0)
    firstLineValues = list(map(int, firstLine.split()[0:2]))

    jobsNb = firstLineValues[0]
    machinesNb = firstLineValues[1]

    jobs = []
    for i in range(jobsNb):
        currentLine = lines[i]
        currentLineValues = list(map(int, currentLine.split()))

        operations = []

        j = 1
        while j < len(currentLineValues):
            k = currentLineValues[j]
            j = j + 1

            operation = []
            for _ in range(k):
                machine = currentLineValues[j]
                j = j + 1
                processingTime = currentLineValues[j]
                j = j + 1

                operation.append({"machine": machine, "processingTime": processingTime})

            operations.append(operation)

        jobs.append(operations)

    return {"machinesNb": machinesNb, "jobs": jobs}


def parse_file(path: Path):
    file = open(path)
    data = file.read()
    file.close()

    return parse(data)


def get_file_data(filename: str) -> tuple:
    path = Path().cwd() / "data" / filename
    info = parse_file(path)

    return get_data(info)


def get_data(info):
    jobs = []
    operations = []
    o_index = 0
    maximum = 0
    for j in info["jobs"]:
        aux_list = []

        for op in j:
            aux_list.append(o_index)
            o_index += 1
            op_aux = [0] * info["machinesNb"]

            for opt in op:
                op_aux[opt["machine"] - 1] = opt["processingTime"]

                if opt["processingTime"] > maximum:
                    maximum = opt["processingTime"]

            operations.append(op_aux)
        jobs.append(aux_list)

    # operations = [[y for y in x] for x in operations]
    return jobs, operations, info, maximum
