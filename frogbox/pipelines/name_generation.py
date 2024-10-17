import random


ADJECTIVES = [
    "adorable",
    "adventurous",
    "aggressive",
    "agreeable",
    "alert",
    "alive",
    "amused",
    "angry",
    "annoyed",
    "annoying",
    "anxious",
    "arrogant",
    "ashamed",
    "attractive",
    "average",
    "awful",
    "bad",
    "beautiful",
    "better",
    "bewildered",
    "black",
    "bloody",
    "blue",
    "blushing",
    "bored",
    "brainy",
    "brave",
    "breakable",
    "bright",
    "busy",
    "calm",
    "careful",
    "cautious",
    "charming",
    "cheerful",
    "clean",
    "clear",
    "clever",
    "cloudy",
    "clumsy",
    "colorful",
    "combative",
    "comfortable",
    "concerned",
    "condemned",
    "confused",
    "cooperative",
    "courageous",
    "crazy",
    "creepy",
    "crowded",
    "cruel",
    "curious",
    "cute",
    "dangerous",
    "dark",
    "dead",
    "defeated",
    "defiant",
    "delightful",
    "depressed",
    "determined",
    "different",
    "difficult",
    "disgusted",
    "distinct",
    "disturbed",
    "dizzy",
    "doubtful",
    "drab",
    "dull",
    "eager",
    "easy",
    "elated",
    "elegant",
    "embarrassed",
    "enchanting",
    "encouraging",
    "energetic",
    "enthusiastic",
    "envious",
    "evil",
    "excited",
    "expensive",
    "exuberant",
    "fair",
    "faithful",
    "famous",
    "fancy",
    "fantastic",
    "fierce",
    "filthy",
    "fine",
    "foolish",
    "fragile",
    "frail",
    "frantic",
    "friendly",
    "frightened",
    "funny",
    "gentle",
    "gifted",
    "glamorous",
    "gleaming",
    "glorious",
    "good",
    "gorgeous",
    "graceful",
    "grieving",
    "grotesque",
    "grumpy",
    "handsome",
    "happy",
    "healthy",
    "helpful",
    "helpless",
    "hilarious",
    "homeless",
    "homely",
    "horrible",
    "hungry",
    "hurt",
    "ill",
    "important",
    "impossible",
    "inexpensive",
    "innocent",
    "inquisitive",
    "itchy",
    "jealous",
    "jittery",
    "jolly",
    "joyous",
    "kind",
    "lazy",
    "light",
    "lively",
    "lonely",
    "long",
    "lovely",
    "lucky",
    "magnificent",
    "misty",
    "modern",
    "motionless",
    "muddy",
    "mushy",
    "mysterious",
    "nasty",
    "naughty",
    "nervous",
    "nice",
    "nutty",
    "obedient",
    "obnoxious",
    "odd",
    "open",
    "outrageous",
    "outstanding",
    "panicky",
    "perfect",
    "plain",
    "pleasant",
    "poised",
    "poor",
    "powerful",
    "precious",
    "prickly",
    "proud",
    "putrid",
    "puzzled",
    "quaint",
    "real",
    "relieved",
    "repulsive",
    "rich",
    "scary",
    "selfish",
    "shiny",
    "shy",
    "silly",
    "sleepy",
    "smiling",
    "smoggy",
    "sore",
    "sparkling",
    "splendid",
    "spotless",
    "stormy",
    "strange",
    "stupid",
    "successful",
    "super",
    "talented",
    "tame",
    "tasty",
    "tender",
    "tense",
    "terrible",
    "thankful",
    "thoughtful",
    "thoughtless",
    "tired",
    "tough",
    "troubled",
    "ugliest",
    "ugly",
    "uninterested",
    "unsightly",
    "unusual",
    "upset",
    "uptight",
    "vast",
    "victorious",
    "vivacious",
    "wandering",
    "weary",
    "wicked",
    "wild",
    "witty",
    "worried",
    "worrisome",
    "wrong",
    "zany",
    "zealous",
]

ANIMALS = [
    "aardvark",
    "albatross",
    "alligator",
    "alpaca",
    "ant",
    "anteater",
    "antelope",
    "ape",
    "armadillo",
    "donkey",
    "baboon",
    "badger",
    "barracuda",
    "bat",
    "bear",
    "beaver",
    "bee",
    "bison",
    "boar",
    "buffalo",
    "butterfly",
    "camel",
    "capybara",
    "caribou",
    "cat",
    "caterpillar",
    "cattle",
    "chamois",
    "cheetah",
    "chicken",
    "chimpanzee",
    "chinchilla",
    "chough",
    "clam",
    "cobra",
    "cockroach",
    "cod",
    "coyote",
    "crab",
    "crane",
    "crocodile",
    "crow",
    "curlew",
    "deer",
    "dinosaur",
    "dog",
    "dogfish",
    "dolphin",
    "dotterel",
    "dove",
    "dragonfly",
    "duck",
    "dugong",
    "dunlin",
    "eagle",
    "echidna",
    "eel",
    "eland",
    "elephant",
    "elk",
    "emu",
    "falcon",
    "ferret",
    "finch",
    "fish",
    "flamingo",
    "fly",
    "fox",
    "frog",
    "gaur",
    "gazelle",
    "gerbil",
    "giraffe",
    "gnat",
    "gnu",
    "goat",
    "goldfinch",
    "goldfish",
    "goose",
    "gorilla",
    "goshawk",
    "grasshopper",
    "grouse",
    "guanaco",
    "gull",
    "hamster",
    "hare",
    "hawk",
    "hedgehog",
    "heron",
    "herring",
    "hippo"
    "hornet",
    "horse",
    "human",
    "hummingbird",
    "hyena",
    "ibex",
    "ibis",
    "jackal",
    "jaguar",
    "jay",
    "jellyfish",
    "kangaroo",
    "kingfisher",
    "koala",
    "kouprey",
    "kudu",
    "lapwing",
    "lark",
    "lemur",
    "leopard",
    "lion",
    "llama",
    "lobster",
    "locust",
    "loris",
    "louse",
    "lyrebird",
    "magpie",
    "mallard",
    "manatee",
    "mandrill",
    "mantis",
    "marten",
    "meerkat",
    "mink",
    "mole",
    "mongoose",
    "monkey",
    "moose",
    "mosquito",
    "mouse",
    "mule",
    "narwhal",
    "newt",
    "octopus",
    "okapi",
    "opossum",
    "oryx",
    "ostrich",
    "otter",
    "owl",
    "oyster",
    "panda",
    "panther",
    "parrot",
    "peafowl",
    "pelican",
    "penguin",
    "pheasant",
    "pig",
    "pigeon",
    "pony",
    "porcupine",
    "porpoise",
    "quail",
    "quelea",
    "quetzal",
    "rabbit",
    "raccoon",
    "rail",
    "ram",
    "rat",
    "raven",
    "reindeer",
    "rhino",
    "rook",
    "salamander",
    "salmon",
    "sandpiper",
    "sardine",
    "scorpion",
    "seahorse",
    "seal",
    "shark",
    "sheep",
    "shrew",
    "skunk",
    "snail",
    "snake",
    "sparrow",
    "spider",
    "spoonbill",
    "squid",
    "squirrel",
    "starling",
    "stingray",
    "stinkbug",
    "stork",
    "swallow",
    "swan",
    "tapir",
    "tarsier",
    "termite",
    "tiger",
    "toad",
    "trout",
    "turkey",
    "turtle",
    "viper",
    "vulture",
    "wallaby",
    "walrus",
    "wasp",
    "weasel",
    "whale",
    "wildcat",
    "wolf",
    "wolverine",
    "wombat",
    "woodcock",
    "woodpecker",
    "worm",
    "wren",
    "yak",
    "zebra",
]


def generate_name() -> str:
    adj = random.choice(ADJECTIVES)
    noun = random.choice(ANIMALS)
    return f"{adj}-{noun}"
