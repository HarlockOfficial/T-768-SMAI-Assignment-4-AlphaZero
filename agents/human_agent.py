import agents.agent as agent


class HumanAgent(agent.Agent):

    def __init__(self, name, params):
        super().__init__(params, f'human_agent_{name}')

    def play(self, game):
        moves = game.generate()
        print(game)
        for i, m in enumerate(moves):
            print(' ' + str(i) + ':', game.get_board().move_to_str(m), end=' ')
        print()
        text = input("Play move number: ")
        while True:
            if text.isnumeric():
                n = int(text)
                if 0 <= n < len(moves):
                    break
            text = input("Play move number: ")
        return moves[n], None
